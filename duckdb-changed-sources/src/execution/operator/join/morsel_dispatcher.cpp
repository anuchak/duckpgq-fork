#include "duckdb/execution/operator/join/morsel_dispatcher.h"

#include "chrono"
#include <iostream>
#include <utility>

namespace duckdb {
    /**
     * Main function of the scheduler that dispatches morsels from the FTable between threads. Or it can
     * be called if a thread could not find a morsel from it's local BFSSharedState.
     *
     * @return - TRUE indicates check the combination value of the state pair, FALSE indicates some
     * work was found hence the state's values can be ignored.
     */

    std::pair<GlobalSSSPState, SSSPLocalState> MorselDispatcher::getBFSMorsel(BaseBFSState *bfsMorsel) {
        switch (globalState) {
            case COMPLETE: {
                return {COMPLETE, NO_WORK_TO_SHARE};
            }
            case IN_PROGRESS_ALL_SRC_SCANNED: {
                return findAvailableSSSP(bfsMorsel);
            }
            case IN_PROGRESS: {
                if (numActiveBFSSharedState < activeBFSSharedState.size()) {
                    mutex.lock();
                    if (numActiveBFSSharedState >= activeBFSSharedState.size()) {
                        mutex.unlock();
                        return findAvailableSSSP(bfsMorsel);
                    }
                    auto duration = std::chrono::system_clock::now().time_since_epoch();
                    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                    auto sourceOffset = nextSourceOffset++;
                    if (sourceOffset == 0) {
                        startTime = millis;
                    }
                    if (sourceOffset >= totalSourcesToLaunch) {
                        globalState =
                                (numActiveBFSSharedState == 0) ? COMPLETE : IN_PROGRESS_ALL_SRC_SCANNED;
                        if (globalState == COMPLETE) {
                            printf("total e2e time %lu ms\n", millis - startTime);
                        }
                        mutex.unlock();
                        return {globalState, NO_WORK_TO_SHARE};
                    }
                    numActiveBFSSharedState++;
                    printf("starting a new bfs source, total: %d | src: %lu | time: %lu\n", numActiveBFSSharedState,
                        sourceOffset, millis);
                    uint32_t newSharedStateIdx = UINT32_MAX;
                    // Find a position for the new SSSP in the list, there are 2 candidates:
                    // 1) the position has a nullptr, add the shared state there
                    // 2) the SSSP is marked MORSEL_COMPLETE, it is complete and can be replaced
                    for (auto i = 0u; i < activeBFSSharedState.size(); i++) {
                        if (!activeBFSSharedState[i]) {
                            newSharedStateIdx = i;
                            break;
                        }
                        activeBFSSharedState[i]->mutex.lock();
                        if (activeBFSSharedState[i]->isComplete()) {
                            newSharedStateIdx = i;
                            activeBFSSharedState[i]->mutex.unlock();
                            break;
                        }
                        activeBFSSharedState[i]->mutex.unlock();
                    }
                    if (newSharedStateIdx == UINT32_MAX || !activeBFSSharedState[newSharedStateIdx]) {
                        auto newBFSSharedState =
                                std::make_shared<BFSSharedState>(upperBound, lowerBound, maxOffset);
                        setUpNewBFSSharedState(newBFSSharedState, bfsMorsel, sourceOffset);
                        newBFSSharedState->startTimeInMillis1 = millis;
                        if (newSharedStateIdx != UINT32_MAX) {
                            activeBFSSharedState[newSharedStateIdx] = newBFSSharedState;
                        }
                    } else {
                        /// HOLD LOCK here for BFSSharedState being reused. While resetting an existing
                        /// BFSSharedState, only this current Thread should have exclusive access to it.
                        activeBFSSharedState[newSharedStateIdx]->mutex.lock();
                        setUpNewBFSSharedState(activeBFSSharedState[newSharedStateIdx], bfsMorsel, sourceOffset);
                        activeBFSSharedState[newSharedStateIdx]->startTimeInMillis1 = millis;
                        activeBFSSharedState[newSharedStateIdx]->mutex.unlock();
                    }
                    mutex.unlock();
                    return {IN_PROGRESS, EXTEND_IN_PROGRESS};
                } else {
                    return findAvailableSSSP(bfsMorsel);
                }
            }
        }
    }

    void MorselDispatcher::setUpNewBFSSharedState(std::shared_ptr<BFSSharedState> &newBFSSharedState,
                                                  BaseBFSState *bfsMorsel, uint64_t srcOffset) {
        newBFSSharedState->reset();
        newBFSSharedState->srcOffset = srcOffset;
        newBFSSharedState->markSrc();
        newBFSSharedState->numThreadsBFSRegistered++;
        bfsMorsel->bfsSharedState = newBFSSharedState.get();
    }

    uint32_t MorselDispatcher::getNextAvailableSSSPWork(BaseBFSState *bfsMorsel) const {
        uint64_t maxBFSWork = 0u, maxBfsID = UINT64_MAX, maxPathLenWriteWork = 0,
                maxPathLenWriteID = UINT64_MAX;
        for (auto i = 0u; i < activeBFSSharedState.size(); i++) {
            if (activeBFSSharedState[i]) {
                auto bfsSharedState = activeBFSSharedState[i];
                if (bfsSharedState->ssspLocalState == EXTEND_IN_PROGRESS &&
                    bfsSharedState->numThreadsBFSFinished == 0u) {
                    auto frontierSize =
                            bfsSharedState->currentFrontierSize -
                            bfsSharedState->nextScanStartIdx.load(std::memory_order::memory_order_acq_rel);
                    auto threadsActive = bfsSharedState->numThreadsBFSRegistered
                                             ? bfsSharedState->numThreadsBFSRegistered
                                             : 1;
                    auto work = frontierSize / threadsActive;
                    if (work > maxBFSWork) {
                        maxBFSWork = work;
                        maxBfsID = i;
                    }
                    continue;
                }
                if (bfsSharedState->ssspLocalState == PATH_LENGTH_WRITE_IN_PROGRESS &&
                    bfsSharedState->numThreadsOutputFinished == 0u) {
                    auto totalOffsets =
                            bfsSharedState->maxOffset -
                            bfsSharedState->nextDstScanStartIdx.load(std::memory_order::memory_order_acq_rel);
                    auto threadsActive = bfsSharedState->numThreadsOutputRegistered
                                             ? bfsSharedState->numThreadsOutputRegistered
                                             : 1;
                    auto work = totalOffsets / threadsActive;
                    if (work > maxPathLenWriteWork) {
                        maxPathLenWriteWork = work;
                        maxPathLenWriteID = i;
                    }
                }
            }
        }
        if (maxBFSWork > 0 && activeBFSSharedState[maxBfsID]->registerThreadForBFS(bfsMorsel)) {
            return maxBfsID;
        }
        if (maxPathLenWriteWork > 0 &&
            activeBFSSharedState[maxPathLenWriteID]->registerThreadForPathOutput()) {
            return maxPathLenWriteID;
        }
        return UINT32_MAX;
    }

    /**
     * Try to find an available BFSSharedState for work and then try to get a morsel. If the work is
     * writing path lengths then the localState value will be PATH_LENGTH_WRITE_IN_PROGRESS.
     */
    std::pair<GlobalSSSPState, SSSPLocalState> MorselDispatcher::findAvailableSSSP(
        BaseBFSState *bfsMorsel) {
        std::pair state{globalState, NO_WORK_TO_SHARE};
        auto availableSSSPIdx = getNextAvailableSSSPWork(bfsMorsel);
        if (availableSSSPIdx == UINT32_MAX) {
            return state;
        }
        bfsMorsel->bfsSharedState = activeBFSSharedState[availableSSSPIdx].get();
        return {globalState, bfsMorsel->bfsSharedState->ssspLocalState};
    }

    /**
     * Return value = -1: New BFSSharedState can be started.
     * Return value = 0, The pathLengthMorsel received did not yield any value to write to the
     * pathLengthVector.
     * Return value > 0, indicates pathLengths were written to pathLengthVector.
     */
    int64_t MorselDispatcher::writeDstNodeIDAndPathLength(std::unique_ptr<BaseBFSState> &baseBfsMorsel) {
        auto bfsSharedState = baseBfsMorsel->bfsSharedState;
        auto startScanIdxAndSize = bfsSharedState->getDstPathLengthMorsel();
        if (startScanIdxAndSize.first == UINT64_MAX && startScanIdxAndSize.second == INT64_MAX) {
            // Don't hold the reference if no work present in this BFS Shared State.
            baseBfsMorsel->bfsSharedState = nullptr;
            if (bfsSharedState->deregisterThreadFromPathOutput()) {
                mutex.lock();
                bfsSharedState->mutex.lock();
                numActiveBFSSharedState--;
                bfsSharedState->ssspLocalState = MORSEL_COMPLETE;
                auto duration = std::chrono::system_clock::now().time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                printf(
                    "Source: %lu | BFS time taken: %lu millis | Path / Path-Len write time taken: %lu "
                    "millis | Total Active BFS: %d\n",
                    bfsSharedState->srcOffset,
                    bfsSharedState->startTimeInMillis2 - bfsSharedState->startTimeInMillis1,
                    millis - bfsSharedState->startTimeInMillis2, numActiveBFSSharedState);
                bfsSharedState->mutex.unlock();
                if (numActiveBFSSharedState == 0 && globalState == IN_PROGRESS_ALL_SRC_SCANNED) {
                    printf("total e2e time %lu ms\n", millis - startTime);
                    globalState = COMPLETE;
                }
                mutex.unlock();
            }
            return -1;
        }
        return baseBfsMorsel->writeToVector(startScanIdxAndSize);
    }
} // namespace duckdb
