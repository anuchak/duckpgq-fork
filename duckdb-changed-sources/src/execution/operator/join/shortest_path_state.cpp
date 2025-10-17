#include <cstdint>

#include "duckdb/execution/operator/join/shortest_path_state.h"

namespace duckdb {

template<>
void ShortestPathState<false>::addToLocalNextBFSLevel(const int64_t *v, std::vector<int64_t> &e,
    unsigned long boundNodeOffset) {
    for (auto start = v[boundNodeOffset]; start < v[boundNodeOffset + 1]; ++start) {
        auto nbrOffset = e[start];
        auto state = bfsSharedState->visitedNodes[nbrOffset];
        if (state == NOT_VISITED_DST) {
            if (__sync_bool_compare_and_swap(&bfsSharedState->visitedNodes[nbrOffset], state, VISITED_DST)) {
                bfsSharedState->pathLength[nbrOffset] = bfsSharedState->currentLevel + 1;
                bfsSharedState->nextFrontier[nbrOffset] = 1u;
                numVisitedDstNodes++;
            }
        } else if (state == NOT_VISITED) {
            __sync_bool_compare_and_swap(&bfsSharedState->visitedNodes[nbrOffset], state, VISITED);
        }
    }
}

template<>
    void ShortestPathState<true>::addToLocalNextBFSLevel(const int64_t *v, std::vector<int64_t> &e,
        unsigned long boundNodeOffset) {

}

template<>
int64_t ShortestPathState<false>::writeToVector(std::pair<uint64_t, int64_t> startScanIdxAndSize) {
    auto size = 0u;
    auto endIdx = startScanIdxAndSize.first + startScanIdxAndSize.second;
    while (startScanIdxAndSize.first < endIdx) {
        if (bfsSharedState->pathLength[startScanIdxAndSize.first] >= bfsSharedState->lowerBound) {
            size++;
        }
        startScanIdxAndSize.first++;
    }
    if (size > 0) {
        return size;
    }
    return 0;
}

template<>
int64_t ShortestPathState<true>::writeToVector(std::pair<uint64_t, int64_t> startScanIdxAndSize) {

}

} // namespace duckdb
