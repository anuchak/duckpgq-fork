#pragma once

#include "bfs_state.h"

namespace duckdb {

template<bool TRACK_PATH>
class ShortestPathState : public BaseBFSState {
public:
    ShortestPathState(uint8_t upperBound, uint8_t lowerBound)
        : BaseBFSState{upperBound, lowerBound}, numVisitedDstNodes{0}, startScanIdx{0u}, endScanIdx{0u} {}

    ~ShortestPathState() override = default;

    inline bool getRecursiveJoinType() final { return TRACK_PATH; }

    inline void resetState() final {
        BaseBFSState::resetState();
        numVisitedDstNodes = 0;
    }

    inline uint64_t getNumVisitedDstNodes() { return numVisitedDstNodes; }
    /// This is used for nTkSCAS scheduler case (no tracking of path + single label case)
    inline void reset(uint64_t startScanIdx_, uint64_t endScanIdx_,
        BFSSharedState* bfsSharedState_) override {
        startScanIdx = startScanIdx_;
        endScanIdx = endScanIdx_;
        bfsSharedState = bfsSharedState_;
        numVisitedDstNodes = 0u;
    }

    // For Shortest Path, multiplicity is always 0
    inline uint64_t getBoundNodeMultiplicity(uint64_t offset) override { return 0u; }

    inline uint64_t getNextNodeOffset() override {
        if (startScanIdx == endScanIdx) {
            return UINT64_MAX;
        }
        if (bfsSharedState->isSparseFrontier) {
            return bfsSharedState->sparseFrontier[startScanIdx++];
        }
        while (startScanIdx != endScanIdx && !bfsSharedState->denseFrontier[startScanIdx]) {
            startScanIdx++;
        }
        if (startScanIdx == endScanIdx) {
            return UINT64_MAX;
        }
        return startScanIdx++;
        // return bfsSharedState->bfsLevelNodeOffsets[startScanIdx++];
    }

    void addToLocalNextBFSLevel(const int64_t *v, std::vector<int64_t> &e, unsigned long boundNodeOffset) override;

    int64_t writeToVector(std::pair<uint64_t, int64_t> startScanIdxAndSize) override;

private:
    uint64_t numVisitedDstNodes;

    /// These will be used for [Single Label, Track None] to track starting, ending index of morsel.
    uint64_t startScanIdx;
    uint64_t endScanIdx;
};

} // namespace duckdb
