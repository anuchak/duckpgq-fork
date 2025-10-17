#pragma once

#include "bfs_state.h"

namespace duckdb {
    struct MorselDispatcher {
    public:
        // upper, lower bound hardcoded to max
        explicit MorselDispatcher(uint64_t totalSourcesToLaunch) : maxOffset{UINT64_MAX}, upperBound{30u},
                                                                   lowerBound{1u}, startTime{0},
                                                                   nextSourceOffset{0u},
                                                                   totalSourcesToLaunch{totalSourcesToLaunch},
                                                                   numActiveBFSSharedState{0u},
                                                                   globalState{IN_PROGRESS} {
        }

        inline void setMaxOffset(uint64_t maxOffset) {
            this->maxOffset = maxOffset;
        }

        inline void initActiveBFSSharedState(uint32_t maxConcurrentBFS) {
            activeBFSSharedState = std::vector<std::shared_ptr<BFSSharedState> >(maxConcurrentBFS, nullptr);
        }

        std::pair<GlobalSSSPState, SSSPLocalState> getBFSMorsel(BaseBFSState *bfsMorsel);

        static void setUpNewBFSSharedState(std::shared_ptr<BFSSharedState> &newBFSSharedState,
                                           BaseBFSState *bfsMorsel, uint64_t srcOffset);

        uint32_t getNextAvailableSSSPWork(BaseBFSState *bfsMorsel) const;

        std::pair<GlobalSSSPState, SSSPLocalState> findAvailableSSSP(BaseBFSState *bfsMorsel);

        int64_t writeDstNodeIDAndPathLength(std::unique_ptr<BaseBFSState> &baseBfsMorsel);

    private:
        uint64_t maxOffset;
        uint64_t upperBound;
        uint64_t lowerBound;
        uint64_t startTime;
        uint64_t nextSourceOffset;
        uint64_t totalSourcesToLaunch;
        std::vector<std::shared_ptr<BFSSharedState> > activeBFSSharedState;
        uint32_t numActiveBFSSharedState;
        GlobalSSSPState globalState;
        std::mutex mutex;
    };
} // namespace duckdb
