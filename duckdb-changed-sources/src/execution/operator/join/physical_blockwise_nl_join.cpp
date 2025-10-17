#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"

#include <duckdb/planner/expression/bound_function_expression.hpp>
#include "../../../../duckpgq-extension/src/include/duckpgq_state.hpp"
#include "../../../../duckpgq-extension/src/include/duckpgq/core/functions/function_data/iterative_length_function_data.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/join/outer_join_marker.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/common/enum_util.hpp"

using namespace duckpgq::core;

namespace duckdb {

PhysicalBlockwiseNLJoin::PhysicalBlockwiseNLJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                                 unique_ptr<PhysicalOperator> right, unique_ptr<Expression> condition,
                                                 JoinType join_type, idx_t estimated_cardinality)
    : PhysicalJoin(op, PhysicalOperatorType::BLOCKWISE_NL_JOIN, join_type, estimated_cardinality),
      condition(std::move(condition)) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	// MARK and SINGLE joins not handled
	D_ASSERT(join_type != JoinType::MARK);
	D_ASSERT(join_type != JoinType::SINGLE);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinLocalState : public LocalSinkState {
public:
	BlockwiseNLJoinLocalState() {
	}
};

class BlockwiseNLJoinGlobalState : public GlobalSinkState {
public:
	explicit BlockwiseNLJoinGlobalState(ClientContext &context, const PhysicalBlockwiseNLJoin &op)
	    : right_chunks(context, op.children[1]->GetTypes()), right_outer(PropagatesBuildSide(op.join_type)) {
		if (context.config.user_variables.find("totalSources") == context.config.user_variables.end()) {
			throw std::runtime_error("totalSources user variable not set");
		}
		morsel_dispatcher = std::make_shared<MorselDispatcher>(
			context.config.user_variables.find("totalSources")->second.GetValue<uint64_t>());
	}

	mutex lock;
	ColumnDataCollection right_chunks;
	std::shared_ptr<MorselDispatcher> morsel_dispatcher;
	OuterJoinMarker right_outer;
};

unique_ptr<GlobalSinkState> PhysicalBlockwiseNLJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<BlockwiseNLJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalBlockwiseNLJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<BlockwiseNLJoinLocalState>();
}

SinkResultType PhysicalBlockwiseNLJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<BlockwiseNLJoinGlobalState>();
	lock_guard<mutex> nl_lock(gstate.lock);
	gstate.right_chunks.Append(chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalBlockwiseNLJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<BlockwiseNLJoinGlobalState>();
	gstate.right_outer.Initialize(gstate.right_chunks.Count());

	if (gstate.right_chunks.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinState : public CachingOperatorState {
public:
	explicit BlockwiseNLJoinState(ExecutionContext &context, ColumnDataCollection &rhs,
	                              const PhysicalBlockwiseNLJoin &op)
	    : op(op), cross_product(rhs), left_outer(IsLeftOuterJoin(op.join_type)), match_sel(STANDARD_VECTOR_SIZE),
	      executor(context.client, *op.condition) {
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
		ResetMatches();
	}

	const PhysicalBlockwiseNLJoin &op;
	CrossProductExecutor cross_product;
	OuterJoinMarker left_outer;
	SelectionVector match_sel;
	ExpressionExecutor executor;
	DataChunk intermediate_chunk;
	std::unique_ptr<BaseBFSState> bfsState;
	bool found_match[STANDARD_VECTOR_SIZE];

	void ResetMatches() {
		if (op.join_type != JoinType::SEMI && op.join_type != JoinType::ANTI) {
			return;
		}
		for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
			found_match[i] = false;
		}
	}
};

unique_ptr<OperatorState> PhysicalBlockwiseNLJoin::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = sink_state->Cast<BlockwiseNLJoinGlobalState>();
	auto result = make_uniq<BlockwiseNLJoinState>(context, gstate.right_chunks, *this);
	result->bfsState = std::make_unique<ShortestPathState<false>>(30u /*upper*/, 1u /*lower*/);
	auto& expressionState = *result->executor.GetStates()[0]->root_state->child_states[0]->child_states[1];
	auto &funcInfo = (BoundFunctionExpression &)expressionState.expr;
	auto &info = static_cast<IterativeLengthFunctionData &>(*funcInfo.bind_info);
	auto lookup = context.client.registered_state->Get<DuckPGQState>("duckpgq");
	auto csrEntry = lookup->csr_list.find((uint64_t) info.csr_id);
	auto totalVertices = csrEntry->second->vsize;
	printf("total vertices in the graph %lu\n", totalVertices);
	gstate.morsel_dispatcher->setMaxOffset(totalVertices);
	gstate.morsel_dispatcher->initActiveBFSSharedState(context.client.db->config.options.maximum_threads);
	info.morsel_dispatcher = gstate.morsel_dispatcher;
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		vector<LogicalType> intermediate_types;
		for (auto &type : children[0]->types) {
			intermediate_types.emplace_back(type);
		}
		for (auto &type : children[1]->types) {
			intermediate_types.emplace_back(type);
		}
		result->intermediate_chunk.Initialize(Allocator::DefaultAllocator(), intermediate_types);
	}
	if (join_type == JoinType::RIGHT_ANTI || join_type == JoinType::RIGHT_SEMI) {
		throw NotImplementedException("physical blockwise RIGHT_SEMI/RIGHT_ANTI join not yet implemented");
	}
	return std::move(result);
}

bool PhysicalBlockwiseNLJoin::scanOutput(OperatorState &state_p) const {
	auto &state = state_p.Cast<BlockwiseNLJoinState>();
	while (true) {
		if (!state.bfsState->hasBFSSharedState()) {
			return false;
		}
		// write values
		auto &gstate = sink_state->Cast<BlockwiseNLJoinGlobalState>();
		auto ret = gstate.morsel_dispatcher->writeDstNodeIDAndPathLength(state.bfsState);
		if (ret > 0) {
			return true;
		}
		if (ret < 0) {
			return false;
		}
	}
}

bool PhysicalBlockwiseNLJoin::computeBFS(ExecutionContext &context, OperatorState &lstate) const {
	auto &gstate = sink_state->Cast<BlockwiseNLJoinGlobalState>();
	auto &block = lstate.Cast<BlockwiseNLJoinState>();
	while (true) {
		auto state = gstate.morsel_dispatcher->getBFSMorsel(block.bfsState.get());
		if (state.first == COMPLETE) {
			return false;
		}
		if (state.second == PATH_LENGTH_WRITE_IN_PROGRESS) {
			return true;
		}
		if (state.second == NO_WORK_TO_SHARE) {
			std::this_thread::sleep_for(std::chrono::microseconds(500));
			if (context.client.interrupted) {
				return false;
			}
			continue;
		}
		while (true) {
			state.second = block.bfsState->getBFSMorsel();
			if (state.second == NO_WORK_TO_SHARE) {
				break;
			}
			if (state.second == PATH_LENGTH_WRITE_IN_PROGRESS) {
				return true;
			}
			computeBFSnThreadkMorsel(lstate, context);
			block.bfsState->finishBFSMorsel();
		}
	}
}

void PhysicalBlockwiseNLJoin::computeBFSnThreadkMorsel(OperatorState &state_p, ExecutionContext &context) {
	auto &state = state_p.Cast<BlockwiseNLJoinState>();
	auto& expressionState = *state.executor.GetStates()[0]->root_state->child_states[0]->child_states[1];
	auto &funcInfo = (BoundFunctionExpression &)expressionState.expr;
	auto &info = static_cast<IterativeLengthFunctionData &>(*funcInfo.bind_info);
	auto lookup = context.client.registered_state->Get<DuckPGQState>("duckpgq");
	auto csrEntry = lookup->csr_list.find((uint64_t) info.csr_id);
	auto nodeOffset = state.bfsState->getNextNodeOffset();
	while (nodeOffset != UINT64_MAX) {
		state.bfsState->addToLocalNextBFSLevel(reinterpret_cast<int64_t *>(csrEntry->second->v),
			csrEntry->second->e, nodeOffset);
		nodeOffset = state.bfsState->getNextNodeOffset();
	}
}

OperatorResultType PhysicalBlockwiseNLJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                            DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                            OperatorState &state_p) const {
	D_ASSERT(input.size() > 0);
	auto &state = state_p.Cast<BlockwiseNLJoinState>();

	while (true) {
		if (scanOutput(state_p)) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		if (!computeBFS(context, state_p)) {
			auto& expressionState = *state.executor.GetStates()[0]->root_state->child_states[0]->child_states[1];
			auto &funcInfo = (BoundFunctionExpression &)expressionState.expr;
			auto &info = static_cast<IterativeLengthFunctionData &>(*funcInfo.bind_info);
			auto duckpgqState = context.client.registered_state->Get<DuckPGQState>("duckpgq");
			duckpgqState->csr_to_delete.insert(info.csr_id);
			return OperatorResultType::FINISHED;
		}
	}

	/*if (gstate.right_chunks.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			PhysicalComparisonJoin::ConstructEmptyJoinResult(join_type, false, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}*/

	/*DataChunk *intermediate_chunk = &chunk;
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		intermediate_chunk = &state.intermediate_chunk;
		intermediate_chunk->Reset();
	}*/


	/*// now perform the actual join
	// we perform a cross product, then execute the expression directly on the cross product result
	idx_t result_count = 0;
	auto result = state.cross_product.Execute(input, *intermediate_chunk);
	if (result == OperatorResultType::NEED_MORE_INPUT) {
		// exhausted input, have to pull new LHS chunk
		if (state.left_outer.Enabled()) {
			// left join: before we move to the next chunk, see if we need to output any vectors that didn't
			// have a match found
			state.left_outer.ConstructLeftJoinResult(input, *intermediate_chunk);
			state.left_outer.Reset();
		}

		if (join_type == JoinType::SEMI) {
			PhysicalJoin::ConstructSemiJoinResult(input, chunk, state.found_match);
		}
		if (join_type == JoinType::ANTI) {
			PhysicalJoin::ConstructAntiJoinResult(input, chunk, state.found_match);
		}
		state.ResetMatches();

		return OperatorResultType::NEED_MORE_INPUT;
	}

	// now perform the computation
	result_count = state.executor.SelectExpression(*intermediate_chunk, state.match_sel);

	// handle anti and semi joins with different logic
	if (result_count > 0) {
		// found a match!
		// handle anti semi join conditions first
		if (join_type == JoinType::ANTI || join_type == JoinType::SEMI) {
			if (state.cross_product.ScanLHS()) {
				state.found_match[state.cross_product.PositionInChunk()] = true;
			} else {
				for (idx_t i = 0; i < result_count; i++) {
					state.found_match[state.match_sel.get_index(i)] = true;
				}
			}
			intermediate_chunk->Reset();
			// trick the loop to continue as semi and anti joins will never produce more output than
			// the LHS cardinality
			result_count = 0;
		} else {
			// check if the cross product is scanning the LHS or the RHS in its entirety
			if (!state.cross_product.ScanLHS()) {
				// set the match flags in the LHS
				state.left_outer.SetMatches(state.match_sel, result_count);
				// set the match flag in the RHS
				gstate.right_outer.SetMatch(state.cross_product.ScanPosition() + state.cross_product.PositionInChunk());
			} else {
				// set the match flag in the LHS
				state.left_outer.SetMatch(state.cross_product.PositionInChunk());
				// set the match flags in the RHS
				gstate.right_outer.SetMatches(state.match_sel, result_count, state.cross_product.ScanPosition());
			}
			intermediate_chunk->Slice(state.match_sel, result_count);
		}
	} else {
		// no result: reset the chunk
		intermediate_chunk->Reset();
	}*/

	return OperatorResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalBlockwiseNLJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = EnumUtil::ToString(join_type);
	result["Condition"] = condition->GetName();
	return result;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinGlobalScanState : public GlobalSourceState {
public:
	explicit BlockwiseNLJoinGlobalScanState(const PhysicalBlockwiseNLJoin &op) : op(op) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
		sink.right_outer.InitializeScan(sink.right_chunks, scan_state);
	}

	const PhysicalBlockwiseNLJoin &op;
	OuterJoinGlobalScanState scan_state;

public:
	idx_t MaxThreads() override {
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
        return sink.right_outer.MaxThreads();
	}
};

class BlockwiseNLJoinLocalScanState : public LocalSourceState {
public:
	explicit BlockwiseNLJoinLocalScanState(const PhysicalBlockwiseNLJoin &op, BlockwiseNLJoinGlobalScanState &gstate) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
	}

	OuterJoinLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalBlockwiseNLJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<BlockwiseNLJoinGlobalScanState>(*this);
}

unique_ptr<LocalSourceState> PhysicalBlockwiseNLJoin::GetLocalSourceState(ExecutionContext &context,
                                                                          GlobalSourceState &gstate) const {
	return make_uniq<BlockwiseNLJoinLocalScanState>(*this, gstate.Cast<BlockwiseNLJoinGlobalScanState>());
}

SourceResultType PhysicalBlockwiseNLJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	D_ASSERT(PropagatesBuildSide(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = sink_state->Cast<BlockwiseNLJoinGlobalState>();
	auto &gstate = input.global_state.Cast<BlockwiseNLJoinGlobalScanState>();
	auto &lstate = input.local_state.Cast<BlockwiseNLJoinLocalScanState>();

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb
