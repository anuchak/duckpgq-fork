#include "duckdb/execution/operator/join/bfs_state.h"
#include "duckdb/execution/operator/join/shortest_path_state.h"

#include <cmath>

#include <immintrin.h>
#include <duckdb/common/vector_size.hpp>

namespace duckdb {

bool BFSSharedState::registerThreadForBFS(BaseBFSState* bfsMorsel) {
    std::unique_lock lock(mutex);
    if (ssspLocalState == MORSEL_COMPLETE) {
        return false;
    }
    if (isBFSComplete()) {
        return false;
    }
    if (ssspLocalState == PATH_LENGTH_WRITE_IN_PROGRESS) {
        return false;
    }
    numThreadsBFSRegistered++;
    return true;
}

bool BFSSharedState::registerThreadForPathOutput() {
    std::unique_lock lock(mutex);
    if (ssspLocalState == MORSEL_COMPLETE || numThreadsOutputFinished != 0u) {
        return false;
    }
    if (ssspLocalState == EXTEND_IN_PROGRESS) {
        return false;
    }
    if (nextDstScanStartIdx.load(std::memory_order::memory_order_acq_rel) >= visitedNodes.size()) {
        return false;
    }
    numThreadsOutputRegistered++;
    return true;
}

bool BFSSharedState::deregisterThreadFromPathOutput() {
    std::unique_lock lock(mutex);
    numThreadsOutputFinished++;
    return (numThreadsOutputRegistered == numThreadsOutputFinished);
}

void BFSSharedState::reset() {
    ssspLocalState = EXTEND_IN_PROGRESS;
    currentLevel = 0u;
    startTimeInMillis1 = 0u;
    startTimeInMillis2 = 0u;
    nextScanStartIdx = 0u;
    numVisitedNodes = 0u;
    // All node offsets are destinations hence mark all as not visited destinations.
    std::fill(visitedNodes.begin(), visitedNodes.end(), NOT_VISITED_DST);
    std::fill(pathLength.begin(), pathLength.end(), 0u);
    // bfsLevelNodeOffsets.clear();
    isSparseFrontier = true;
    nextFrontierSize = 0u;
    if (!nextFrontier) {
        sparseFrontier = std::vector<uint64_t>();
        sparseFrontier.reserve(maxOffset + 1);
        // skip allocating dense frontier unless required
        nextFrontier = new uint8_t[maxOffset + 1];
        std::fill(nextFrontier, nextFrontier + maxOffset + 1, 0u);
    } else {
        sparseFrontier.clear();
        if (denseFrontier) {
            std::fill(denseFrontier, denseFrontier + maxOffset + 1, 0u);
        }
        std::fill(nextFrontier, nextFrontier + maxOffset + 1, 0u);
    }
    srcOffset = 0u;
    numThreadsBFSRegistered = 0u;
    numThreadsOutputRegistered = 0u;
    numThreadsBFSFinished = 0u;
    numThreadsOutputFinished = 0u;
    nextDstScanStartIdx = 0u;
}

SSSPLocalState BFSSharedState::getBFSMorsel(BaseBFSState* bfsMorsel) {
    auto morselStartIdx =
        nextScanStartIdx.fetch_add(bfsMorsel->bfsMorselSize * sparseFactor, std::memory_order_acq_rel);
    if (morselStartIdx >= currentFrontierSize ||
        isBFSComplete()) {
        mutex.lock();
        numThreadsBFSFinished++;
        if (numThreadsBFSRegistered != numThreadsBFSFinished) {
            mutex.unlock();
            bfsMorsel->bfsSharedState = nullptr;
            return NO_WORK_TO_SHARE;
        }
        if (isBFSComplete()) {
            numThreadsBFSRegistered = 0, numThreadsBFSFinished = 0;
            auto duration = std::chrono::system_clock::now().time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            startTimeInMillis2 = millis;
            ssspLocalState = PATH_LENGTH_WRITE_IN_PROGRESS;
            numThreadsOutputRegistered++;
            mutex.unlock();
            return PATH_LENGTH_WRITE_IN_PROGRESS;
        }
        moveNextLevelAsCurrentLevel();
        numThreadsBFSRegistered = 0, numThreadsBFSFinished = 0;
        if (isBFSComplete()) {
            auto duration = std::chrono::system_clock::now().time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            startTimeInMillis2 = millis;
            ssspLocalState = PATH_LENGTH_WRITE_IN_PROGRESS;
            numThreadsOutputRegistered++;
            mutex.unlock();
            return PATH_LENGTH_WRITE_IN_PROGRESS;
        }
        numThreadsBFSRegistered++;
        morselStartIdx =
            nextScanStartIdx.fetch_add(bfsMorsel->bfsMorselSize * sparseFactor, std::memory_order_acq_rel);
        mutex.unlock();
    }
    uint64_t morselEndIdx =
        std::min(morselStartIdx + bfsMorsel->bfsMorselSize * sparseFactor, currentFrontierSize);
    bfsMorsel->reset(morselStartIdx, morselEndIdx, this);
    return EXTEND_IN_PROGRESS;
}

void BFSSharedState::finishBFSMorsel(BaseBFSState* bfsMorsel) {
    auto shortestPathMorsel = reinterpret_cast<ShortestPathState<false>*>(bfsMorsel);
    numVisitedNodes.fetch_add(shortestPathMorsel->getNumVisitedDstNodes(),
        std::memory_order_acq_rel);
    nextFrontierSize.fetch_add(shortestPathMorsel->getNumVisitedDstNodes());
}

bool BFSSharedState::isBFSComplete() const {
    if (currentFrontierSize == 0u) { // no more to extend.
        return true;
    }
    if (currentLevel == upperBound) { // upper limit reached.
        return true;
    }
    return numVisitedNodes == maxOffset;
}

void BFSSharedState::markSrc() {
    currentFrontierSize = 1u;
    visitedNodes[srcOffset] = VISITED_DST;
    ++numVisitedNodes;
    pathLength[srcOffset] = 0;
    isSparseFrontier = true;
    sparseFrontier.push_back(srcOffset);
}

void BFSSharedState::moveNextLevelAsCurrentLevel() {
    /*auto duration = std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();*/
    currentLevel++;
    nextScanStartIdx = 0u;
    currentFrontierSize = 0u;
    if (currentLevel < upperBound && nextFrontierSize > 0) {
        currentFrontierSize = nextFrontierSize;
        nextFrontierSize = 0u;
        if (currentFrontierSize < (uint64_t)std::ceil(maxOffset / 8)) {
            isSparseFrontier = true;
            sparseFactor = 1u;
            sparseFrontier.clear();
            auto simdWidth = 16u, i = 0u, pos = 0u;
            // SSE2 vector with all elements set to 1
            __m128i ones = _mm_set1_epi8(1);
            for (; i + simdWidth < maxOffset + 1 && pos < currentFrontierSize; i += simdWidth) {
                __m128i vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&nextFrontier[i]));
                __m128i cmp = _mm_cmpeq_epi8(vec, ones);
                int mask = _mm_movemask_epi8(cmp);
                while (mask != 0) {
                    int index = __builtin_ctz(mask);
                    sparseFrontier[pos++] = i + index;
                    mask &= ~(1 << index);
                }
            }

            // Process any remaining elements
            for (; i < maxOffset + 1 && pos < currentFrontierSize; ++i) {
                if (nextFrontier[i]) {
                    sparseFrontier[pos++] = i;
                }
            }
        } else {
            auto actualFrontierSize = currentFrontierSize;
            currentFrontierSize = maxOffset + 1;
            sparseFactor =
                std::ceil( (double) currentFrontierSize / (double) actualFrontierSize);
            isSparseFrontier = false;
            if (!denseFrontier) {
                denseFrontier = new uint8_t[currentFrontierSize];
                std::fill(denseFrontier, denseFrontier + maxOffset + 1, 0u);
            }
            auto temp = denseFrontier;
            denseFrontier = nextFrontier;
            nextFrontier = temp;
        }
        std::fill(nextFrontier, nextFrontier + maxOffset + 1, 0u);
    }
    /*auto duration1 = std::chrono::system_clock::now().time_since_epoch();
    auto millis1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
    printf("time taken to move level %d is %lu ms\n", currentLevel, millis1 - millis);*/
}

std::pair<uint64_t, int64_t> BFSSharedState::getDstPathLengthMorsel() {
    auto morselStartIdx =
        nextDstScanStartIdx.fetch_add((uint64_t) DEFAULT_STANDARD_VECTOR_SIZE, std::memory_order_acq_rel);
    if (morselStartIdx >= visitedNodes.size()) {
        return {UINT64_MAX, INT64_MAX};
    }
    uint64_t morselSize =
        std::min((uint64_t) DEFAULT_STANDARD_VECTOR_SIZE, visitedNodes.size() - morselStartIdx);
    return {morselStartIdx, morselSize};
}

} // namespace duckdb
