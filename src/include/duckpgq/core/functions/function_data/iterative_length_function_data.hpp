//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/function_data/iterative_length_function_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include <duckdb/execution/operator/join/morsel_dispatcher.h>

#include "duckdb/main/client_context.hpp"
#include "duckpgq/common.hpp"

namespace duckpgq {
namespace core {

struct IterativeLengthFunctionData final : FunctionData {
  ClientContext &context;
  int32_t csr_id;
  std::shared_ptr<MorselDispatcher> morsel_dispatcher;

  IterativeLengthFunctionData(ClientContext &context, int32_t csr_id)
      : context(context), csr_id(csr_id) {}
  static unique_ptr<FunctionData>
  IterativeLengthBind(ClientContext &context, ScalarFunction &bound_function,
                      vector<unique_ptr<Expression>> &arguments);

  unique_ptr<FunctionData> Copy() const override;
  bool Equals(const FunctionData &other_p) const override;
};

} // namespace core
} // namespace duckpgq
