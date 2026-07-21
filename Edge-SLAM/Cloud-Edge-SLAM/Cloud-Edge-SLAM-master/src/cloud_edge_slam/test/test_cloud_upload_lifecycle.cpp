#define CLOUD_EDGE_UPLOAD_LIFECYCLE_ONLY
#include "../src/cloud_edge.cpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

struct FakeTicketResult {
    bool accepted = false;
};

int main() {
    using cloud_edge_slam_sla::ActiveUpload;
    using cloud_edge_slam_sla::GetActiveUploadCountForTesting;
    using cloud_edge_slam_sla::IsAcceptingUploadsForTesting;
    using cloud_edge_slam_sla::StopAcceptingUploads;
    using cloud_edge_slam_sla::TryStartUpload;
    using cloud_edge_slam_sla::UploadLifecycleState;
    using cloud_edge_slam_sla::WaitForActiveUploads;

    std::shared_ptr<UploadLifecycleState> state =
        std::make_shared<UploadLifecycleState>();
    std::atomic<bool> drainFinished(false);
    std::thread drainThread;

    {
        ActiveUpload activeUpload = TryStartUpload(state);
        assert(activeUpload);
        assert(GetActiveUploadCountForTesting(state) == 1U);

        StopAcceptingUploads(state);
        assert(!IsAcceptingUploadsForTesting(state));

        ActiveUpload rejectedUpload = TryStartUpload(state);
        assert(!rejectedUpload);
        assert(GetActiveUploadCountForTesting(state) == 1U);

        drainThread = std::thread([state, &drainFinished]() {
            WaitForActiveUploads(state);
            drainFinished.store(true, std::memory_order_release);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        assert(!drainFinished.load(std::memory_order_acquire));
    }

    drainThread.join();
    assert(drainFinished.load(std::memory_order_acquire));
    assert(GetActiveUploadCountForTesting(state) == 0U);

    std::vector<int> tickets;
    tickets.push_back(11);
    tickets.push_back(12);
    tickets.push_back(13);
    int waitedTicketCount = 0;
    cloud_edge_slam_sla::TicketBatchResult<FakeTicketResult> batch =
        cloud_edge_slam_sla::WaitForTicketBatch<int, FakeTicketResult>(
            tickets,
            [&waitedTicketCount](const int ticket, FakeTicketResult &result) {
                waitedTicketCount++;
                result.accepted = ticket != 12;
                return true;
            },
            [](const FakeTicketResult &result) {
                return result.accepted;
            });
    assert(waitedTicketCount == 3);
    assert(batch.results.size() == 3U);
    assert(!batch.succeeded);

    std::vector<int> noTickets;
    batch = cloud_edge_slam_sla::WaitForTicketBatch<int, FakeTicketResult>(
        noTickets,
        [](const int, FakeTicketResult &) {
            return true;
        },
        [](const FakeTicketResult &) {
            return true;
        });
    assert(!batch.succeeded);
    return 0;
}
