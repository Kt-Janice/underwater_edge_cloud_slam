#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace {

std::string ReadFile(const std::string &path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string ExtractFunction(
    const std::string &source,
    const std::string &signature) {
    const std::size_t signaturePosition = source.find(signature);
    if (signaturePosition == std::string::npos) {
        return std::string();
    }

    const std::size_t bodyPosition = source.find('{', signaturePosition);
    if (bodyPosition == std::string::npos) {
        return std::string();
    }

    int depth = 0;
    for (std::size_t index = bodyPosition; index < source.size(); index++) {
        if (source[index] == '{') {
            depth++;
        } else if (source[index] == '}') {
            depth--;
            if (depth == 0) {
                return source.substr(
                    signaturePosition,
                    index - signaturePosition + 1);
            }
        }
    }

    return std::string();
}

bool Require(
    const bool condition,
    const std::string &message) {
    if (condition) {
        return true;
    }

    std::cerr << "FAIL: " << message << std::endl;
    return false;
}

enum class TestCleanupAction {
    NONE,
    DELETE_MAP
};

enum class TestOutcome {
    COMPLETED,
    PARTIAL_FAILURE,
    FAILED_EXCEPTION
};

struct FakeMap {
    FakeMap(
        const unsigned long frontMapId,
        const unsigned long backMapId,
        FakeMap **pCurrentAlias,
        bool *pAliasWasCleared)
        : edgeFrontMapId(frontMapId),
          edgeBackMapId(backMapId),
          ppCurrentAlias(pCurrentAlias),
          pAliasWasCleared(pAliasWasCleared) {
    }

    ~FakeMap() {
        destructionCount++;
        if (ppCurrentAlias != nullptr && pAliasWasCleared != nullptr &&
            *ppCurrentAlias == nullptr) {
            *pAliasWasCleared = true;
        }
    }

    static int destructionCount;
    unsigned long edgeFrontMapId;
    unsigned long edgeBackMapId;
    FakeMap **ppCurrentAlias;
    bool *pAliasWasCleared;
};

int FakeMap::destructionCount = 0;

class FakeTicket {
public:
    bool TryComplete(const TestOutcome outcome) {
        attemptCount++;
        if (completed) {
            return false;
        }

        completed = true;
        completionCount++;
        completedOutcome = outcome;
        return true;
    }

    bool completed = false;
    int attemptCount = 0;
    int completionCount = 0;
    TestOutcome completedOutcome = TestOutcome::FAILED_EXCEPTION;
};

struct FakePendingMap {
    FakeMap *pMap = nullptr;
    FakeTicket *pTicket = nullptr;
};

struct TestExecutionResult {
    TestOutcome outcome = TestOutcome::FAILED_EXCEPTION;
    TestCleanupAction cleanupAction = TestCleanupAction::NONE;
};

void ApplyFakeCleanup(
    FakePendingMap &pending,
    const TestCleanupAction cleanupAction) {
    if (cleanupAction == TestCleanupAction::DELETE_MAP &&
        pending.pMap != nullptr) {
        delete pending.pMap;
        pending.pMap = nullptr;
    }
}

void DispatchFakeMerge(
    FakePendingMap &pending,
    FakeMap *&pCurrentAlias,
    const std::function<TestExecutionResult(FakeMap &)> &strategy) {
    pCurrentAlias = pending.pMap;
    TestExecutionResult result;
    try {
        result = strategy(*pending.pMap);
    } catch (...) {
        result.outcome = TestOutcome::FAILED_EXCEPTION;
        result.cleanupAction = TestCleanupAction::DELETE_MAP;
    }

    pCurrentAlias = nullptr;
    ApplyFakeCleanup(pending, result.cleanupAction);
    pending.pTicket->TryComplete(result.outcome);
}

bool IsPendingMapRelatedToActiveMap(
    const FakeMap &pendingMap,
    const unsigned long activeMapId) {
    if (pendingMap.edgeFrontMapId == activeMapId) {
        return true;
    }
    if (pendingMap.edgeBackMapId == activeMapId) {
        return true;
    }
    return false;
}

bool ShouldWaitForReset(
    const bool resetPending,
    const bool finishRequested,
    const bool finished) {
    if (!resetPending || finishRequested || finished) {
        return false;
    }
    return true;
}

class FakeLocalMapper {
public:
    void RequestStop() {
        stopRequested = true;
    }

    bool isStopped() const {
        return stopped;
    }

    bool isFinished() const {
        return finished;
    }

    void EmptyQueue() {
        emptyQueueCount++;
    }

    void Release() {
        releaseCount++;
    }

    bool stopRequested = false;
    bool stopped = true;
    bool finished = false;
    int emptyQueueCount = 0;
    int releaseCount = 0;
};

class FakeLocalMapperPauseGuard {
public:
    explicit FakeLocalMapperPauseGuard(FakeLocalMapper &localMapper)
        : mLocalMapper(localMapper) {
    }

    ~FakeLocalMapperPauseGuard() {
        Release();
    }

    bool PauseAndDrain() {
        mLocalMapper.RequestStop();
        releaseRequired = true;
        if (mLocalMapper.isFinished()) {
            releaseRequired = false;
            return false;
        }
        if (!mLocalMapper.isStopped()) {
            return false;
        }
        mLocalMapper.EmptyQueue();
        return true;
    }

    void Release() {
        if (!releaseRequired) {
            return;
        }
        releaseRequired = false;
        if (!mLocalMapper.isFinished()) {
            mLocalMapper.Release();
        }
    }

private:
    FakeLocalMapper &mLocalMapper;
    bool releaseRequired = false;
};

bool RunExecutableOwnershipTests() {
    bool passed = true;
    FakeMap::destructionCount = 0;

    FakeMap *currentAlias = nullptr;
    bool exceptionAliasWasCleared = false;
    FakeTicket exceptionTicket;
    FakePendingMap exceptionPending;
    exceptionPending.pMap = new FakeMap(
        10,
        11,
        &currentAlias,
        &exceptionAliasWasCleared);
    exceptionPending.pTicket = &exceptionTicket;
    DispatchFakeMerge(
        exceptionPending,
        currentAlias,
        [](FakeMap &) -> TestExecutionResult {
            throw std::runtime_error("injected strategy failure");
        });

    passed &= Require(
        exceptionPending.pMap == nullptr,
        "exception path must relinquish the pending ownership pointer");
    passed &= Require(
        FakeMap::destructionCount == 1,
        "exception path must destroy its map exactly once");
    passed &= Require(
        exceptionAliasWasCleared,
        "current alias must be cleared before exception cleanup deletes map");
    passed &= Require(
        exceptionTicket.completionCount == 1 &&
            exceptionTicket.completedOutcome == TestOutcome::FAILED_EXCEPTION,
        "exception path must complete its ticket exactly once");
    passed &= Require(
        !exceptionTicket.TryComplete(TestOutcome::COMPLETED) &&
            exceptionTicket.completionCount == 1,
        "ticket must reject a second completion");

    bool partialAliasWasCleared = false;
    FakeTicket partialTicket;
    FakePendingMap partialPending;
    partialPending.pMap = new FakeMap(
        20,
        21,
        &currentAlias,
        &partialAliasWasCleared);
    partialPending.pTicket = &partialTicket;
    DispatchFakeMerge(
        partialPending,
        currentAlias,
        [](FakeMap &) {
            TestExecutionResult result;
            result.outcome = TestOutcome::PARTIAL_FAILURE;
            result.cleanupAction = TestCleanupAction::DELETE_MAP;
            return result;
        });

    passed &= Require(
        partialPending.pMap == nullptr &&
            FakeMap::destructionCount == 2 &&
            partialAliasWasCleared,
        "partial failure must explicitly delete the unclaimed cloud container");
    passed &= Require(
        partialTicket.completionCount == 1 &&
            partialTicket.completedOutcome == TestOutcome::PARTIAL_FAILURE,
        "partial failure must preserve its outcome while completing once");

    FakeMap relatedByFront(31, 32, nullptr, nullptr);
    FakeMap relatedByBack(40, 41, nullptr, nullptr);
    FakeMap unrelated(50, 51, nullptr, nullptr);
    passed &= Require(
        IsPendingMapRelatedToActiveMap(relatedByFront, 31) &&
            IsPendingMapRelatedToActiveMap(relatedByBack, 41) &&
            !IsPendingMapRelatedToActiveMap(unrelated, 60),
        "active reset must match pending maps by both edge map ids");
    passed &= Require(
        ShouldWaitForReset(true, false, false) &&
            !ShouldWaitForReset(true, true, false) &&
            !ShouldWaitForReset(true, false, true),
        "reset wait predicate must unblock on finish request or completion");

    FakeLocalMapper localMapper;
    try {
        FakeLocalMapperPauseGuard pauseGuard(localMapper);
        passed &= Require(
            pauseGuard.PauseAndDrain(),
            "local mapper pause guard must stop and drain an active mapper");
        throw std::runtime_error("injected merge exception");
    } catch (const std::runtime_error &) {
    }
    passed &= Require(
        localMapper.stopRequested &&
            localMapper.emptyQueueCount == 1 &&
            localMapper.releaseCount == 1,
        "local mapper pause guard must release on exceptional exit");

    FakeLocalMapper finishedLocalMapper;
    finishedLocalMapper.finished = true;
    {
        FakeLocalMapperPauseGuard pauseGuard(finishedLocalMapper);
        passed &= Require(
            !pauseGuard.PauseAndDrain(),
            "finished LocalMapper must abort the pause wait");
    }
    passed &= Require(
        finishedLocalMapper.releaseCount == 0,
        "finished LocalMapper must not be released again");

    return passed;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0]
                  << " <CloudMerging_SLA.cc>" << std::endl;
        return 2;
    }

    const std::string source = ReadFile(argv[1]);
    bool passed = Require(!source.empty(), "SLA source must be readable");

    const std::string runBody = ExtractFunction(
        source,
        "void CloudMerging::Run(");
    const std::string seaBody = ExtractFunction(
        source,
        "CloudMergeExecutionResult CloudMerging::RunSeaCloudMerge(");
    const std::string landAirBody = ExtractFunction(
        source,
        "CloudMergeExecutionResult CloudMerging::RunLandAirCloudMerge(");
    const std::string cleanupBody = ExtractFunction(
        source,
        "void CloudMerging::ApplyCloudMapCleanup(");
    const std::string insertionBody = ExtractFunction(
        source,
        "CloudMergeTicketPtr CloudMerging::InsertCloudMapWithTicket(");
    const std::string cancellationBody = ExtractFunction(
        source,
        "void CloudMerging::CancelPendingCloudMaps(");

    passed &= Require(!runBody.empty(), "public Run() must exist");
    passed &= Require(!seaBody.empty(), "sea strategy must exist");
    passed &= Require(!landAirBody.empty(), "land/air strategy must exist");
    passed &= Require(!cleanupBody.empty(), "cleanup function must exist");
    passed &= Require(!insertionBody.empty(), "ticket insertion must exist");
    passed &= Require(!cancellationBody.empty(), "queue cancellation must exist");

    passed &= Require(
        seaBody.find("RunLandAirCloudMerge") == std::string::npos,
        "sea strategy must not delegate to land/air strategy");
    passed &= Require(
        runBody.find(
            "ApplyCloudMapCleanup(pending, executionResult.cleanupAction)") !=
            std::string::npos,
        "public Run() must consume the strategy cleanup action");
    passed &= Require(
        runBody.find("delete pending.pMap") == std::string::npos,
        "public Run() must not infer deletion from an outcome");
    passed &= Require(
        cleanupBody.find(
            "cleanupAction == CloudMapCleanupAction::DELETE_CLOUD_MAP") !=
            std::string::npos,
        "cleanup must require an explicit DELETE_CLOUD_MAP action");
    passed &= Require(
        cleanupBody.find("pending.pMap != nullptr") != std::string::npos,
        "cleanup must require a non-null pending map");
    passed &= Require(
        cleanupBody.find("delete pending.pMap") != std::string::npos,
        "cleanup must delete through the pending ownership pointer");
    passed &= Require(
        cleanupBody.find("pending.pMap = nullptr") != std::string::npos,
        "cleanup must clear ownership after deletion");
    passed &= Require(
        seaBody.find("delete mpCurrentCloudMap") == std::string::npos &&
            landAirBody.find("delete mpCurrentCloudMap") == std::string::npos,
        "strategies must not delete through the non-owning current-map alias");
    passed &= Require(
        insertionBody.find("CloudMapCleanupAction::DELETE_CLOUD_MAP") !=
            std::string::npos,
        "shutdown-rejected insertion must explicitly release its map");
    passed &= Require(
        cancellationBody.find("CloudMapCleanupAction::DELETE_CLOUD_MAP") !=
            std::string::npos,
        "cancelled queued maps must use explicit ownership cleanup");
    passed &= Require(
        seaBody.find("catch (const std::exception &exception)") !=
                std::string::npos &&
            landAirBody.find("catch (const std::exception &exception)") !=
                std::string::npos,
        "both strategies must map exceptions to explicit outcomes");
    passed &= Require(
        seaBody.find("CanReplayDeferredBuffers()") != std::string::npos,
        "sea replay must validate wrapper replay capability");
    passed &= Require(
        source.find("pending->pMap->edgeFrontMapMnId == activeMapId") !=
                std::string::npos &&
            source.find("pending->pMap->edgeBackMapMnId == activeMapId") !=
                std::string::npos,
        "active reset must cancel pending maps associated by edge ids");
    passed &= Require(
        source.find("LocalMapperPauseGuard") != std::string::npos &&
            source.find("PauseAndDrain()") != std::string::npos,
        "LocalMapper stop/release must use an exception-safe guard");

    const std::size_t cleanupCall = runBody.find(
        "ApplyCloudMapCleanup(pending, executionResult.cleanupAction)");
    const std::size_t clearCurrentMap = runBody.find(
        "mpCurrentCloudMap = nullptr");
    passed &= Require(
        cleanupCall != std::string::npos &&
            clearCurrentMap != std::string::npos &&
            clearCurrentMap < cleanupCall,
        "public Run() must clear current-map aliases before cleanup");

    passed &= RunExecutableOwnershipTests();

    if (!passed) {
        return 1;
    }

    std::cout << "PASS: CloudMerge outcome/ownership contract" << std::endl;
    return 0;
}
