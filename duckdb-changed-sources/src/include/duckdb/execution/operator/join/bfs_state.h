#pragma once

#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <unordered_map>

namespace duckdb {

/**
 * States used for nTkS scheduler to mark different status of node offsets.
 * VISITED_NEW and VISITED_DST_NEW are the states when a node is initially visited and
 * when we scan to prepare the bfsLevelNodes vector for next level extension,
 * VISITED_NEW -> VISITED | VISITED_DST_NEW -> VISITED_DST
 */
enum VisitedState : uint8_t {
    NOT_VISITED_DST = 0,
    VISITED_DST = 1,
    NOT_VISITED = 2,
    VISITED = 3,
};

/**
 * The Lifecycle of an BFSSharedState in nTkS scheduler are the 3 following states. Depending on
 * which state an BFSSharedState is in, the thread will held that state :-
 *
 * 1) EXTEND_IN_PROGRESS: thread helps in level extension, try to get a morsel if available
 *
 * 2) PATH_LENGTH_WRITE_IN_PROGRESS: thread helps in writing path length to vector
 *
 * 3) MORSEL_COMPLETE: morsel is complete, try to launch another BFSSharedState (if available),
 *                     else find an existing one
 */
enum SSSPLocalState {
    EXTEND_IN_PROGRESS,
    PATH_LENGTH_WRITE_IN_PROGRESS,
    MORSEL_COMPLETE,

    // NOTE: This is an intermediate state returned ONLY when no work could be provided to a thread.
    // It will not be assigned to the local SSSPLocalState inside a BFSSharedState.
    NO_WORK_TO_SHARE
};

/**
 * Global states of MorselDispatcher used primarily for nTkS scheduler :-
 *
 * 1) IN_PROGRESS: Globally (across all threads active), there is still work available.
 * BFSSharedStates are available for threads to launch.
 *
 * 2) IN_PROGRESS_ALL_SRC_SCANNED: All BFSSharedState available to be launched have been launched.
 * It indicates that the inputFTable has no more tuples to be scanned. The threads have to search
 * for an active BFSSharedState (either in EXTEND_IN_PROGRESS / PATH_LENGTH_WRITE_IN_PROGRESS) to
 * execute.
 *
 * 3) COMPLETE: All BFSSharedStates have been completed, there is no more work either in BFS
 * extension or path length writing to vector. Threads can exit completely (return false and return
 * to parent operator).
 */
enum GlobalSSSPState { IN_PROGRESS, IN_PROGRESS_ALL_SRC_SCANNED, COMPLETE };

struct BaseBFSState;

/**
 * A BFSSharedState is a unit of work for the nTkS scheduling policy. It is *ONLY* used for the
 * particular case of SINGLE_LABEL | NO_PATH_TRACKING. A BFSSharedState is *NOT*
 * exclusive to a thread and any thread can pick up BFS extension or Writing Path Length.
 * A shared_ptr is maintained by the BaseBFSMorsel and a morsel of work is fetched using this ptr.
 */
struct BFSSharedState {
public:
    BFSSharedState(uint64_t upperBound_, uint64_t lowerBound_, uint64_t maxNodeOffset_)
        : ssspLocalState{EXTEND_IN_PROGRESS}, sparseFactor{1u},
          currentLevel{0u}, nextScanStartIdx{0u}, numVisitedNodes{0u},
          visitedNodes{std::vector<uint8_t>(maxNodeOffset_ + 1, NOT_VISITED)},
          pathLength{std::vector<uint8_t>(maxNodeOffset_ + 1, 0u)}, currentFrontierSize{0u},
          nextFrontierSize{0u}, denseFrontier{nullptr}, nextFrontier{nullptr}, srcOffset{0u},
          maxOffset{maxNodeOffset_}, upperBound{upperBound_}, lowerBound{lowerBound_},
          numThreadsBFSRegistered{0u}, numThreadsBFSFinished{0u}, numThreadsOutputRegistered{0u},
          numThreadsOutputFinished{0u}, nextDstScanStartIdx{0u} {}

    ~BFSSharedState() {
        if (nextFrontier) {
            delete[] nextFrontier;
        }
        if (denseFrontier) {
            delete[] denseFrontier;
        }
    }

    bool registerThreadForBFS(BaseBFSState* bfsMorsel);

    bool registerThreadForPathOutput();

    bool deregisterThreadFromPathOutput();

    inline bool isComplete() const { return ssspLocalState == MORSEL_COMPLETE; }

    void reset();

    SSSPLocalState getBFSMorsel(BaseBFSState* bfsMorsel);

    SSSPLocalState getBFSMorselAdaptive(BaseBFSState* bfsMorsel);

    void finishBFSMorsel(BaseBFSState* bfsMorsel);

    // If BFS has completed.
    bool isBFSComplete() const;
    // Mark src as visited.
    void markSrc();

    void moveNextLevelAsCurrentLevel();

    std::pair<uint64_t, int64_t> getDstPathLengthMorsel();

public:
    std::mutex mutex;
    SSSPLocalState ssspLocalState;
    uint64_t startTimeInMillis1;
    uint64_t startTimeInMillis2;
    uint64_t sparseFactor;
    uint8_t currentLevel;
    char padding0[64];
    std::atomic<uint64_t> nextScanStartIdx;
    char padding1[64];

    // Visited state
    std::atomic<uint64_t> numVisitedNodes;
    char padding2[64];
    std::vector<uint8_t> visitedNodes;
    std::vector<uint8_t> pathLength;

    uint64_t currentFrontierSize;
    char padding3[64];
    std::atomic<uint64_t> nextFrontierSize;
    char padding4[64];
    bool isSparseFrontier;
    // sparse frontier
    std::vector<uint64_t> sparseFrontier;
    // dense frontier
    uint8_t* denseFrontier;
    // next frontier
    uint8_t* nextFrontier;

    // Offset of src node.
    uint64_t srcOffset;
    // Maximum offset of dst nodes.
    uint64_t maxOffset;
    uint64_t upperBound;
    uint64_t lowerBound;
    uint32_t numThreadsBFSRegistered;
    uint32_t numThreadsBFSFinished;
    uint32_t numThreadsOutputRegistered;
    uint32_t numThreadsOutputFinished;
    char padding5[64];
    std::atomic<uint64_t> nextDstScanStartIdx;
    char padding6[64];
};

class BaseBFSState {
public:
    explicit BaseBFSState(uint8_t upperBound, uint8_t lowerBound)
        : upperBound{upperBound}, lowerBound{lowerBound}, currentLevel{0}, nextNodeIdxToExtend{0}, bfsMorselSize{256u},
    bfsSharedState{nullptr} {}

    virtual ~BaseBFSState() = default;

    virtual bool getRecursiveJoinType() = 0;

    bool hasBFSSharedState() const { return bfsSharedState != nullptr; }

    virtual void resetState() {
        currentLevel = 0;
        nextNodeIdxToExtend = 0;
    }

    inline SSSPLocalState getBFSMorsel() {
        return bfsSharedState->getBFSMorsel(this);
    }

    inline void finishBFSMorsel() {
        bfsSharedState->finishBFSMorsel(this);
    }

    /// This is used for nTkSCAS scheduler case (no tracking of path + single label case)
    virtual void reset(uint64_t startScanIdx, uint64_t endScanIdx,
        BFSSharedState* bfsSharedState) = 0;

    virtual uint64_t getBoundNodeMultiplicity(uint64_t nodeOffset) = 0;

    virtual void addToLocalNextBFSLevel(const int64_t *v, std::vector<int64_t> &e, unsigned long boundNodeOffset) = 0;

    virtual uint64_t getNextNodeOffset() = 0;

    virtual int64_t writeToVector(std::pair<uint64_t, int64_t> startScanIdxAndSize) = 0;

protected:
    // Static information
    uint8_t upperBound;
    uint8_t lowerBound;
    // Level state
    uint8_t currentLevel;
    uint64_t nextNodeIdxToExtend; // next node to extend from current frontier.

public:
    uint64_t bfsMorselSize;
    BFSSharedState* bfsSharedState;
};

} // namespace duckdb
