#include <gtest/gtest.h>

#include <thread>

#include "CloudMergeTicket.h"

namespace ORB_SLAM3 {

class CloudMergeTicketTestAccess {
public:
    static bool Complete(
        CloudMergeTicket &ticket,
        const CloudMergeResult &result) {
        return ticket.TryComplete(result);
    }

    static bool Wait(
        const CloudMergeTicket &ticket,
        CloudMergeResult &result) {
        return ticket.Wait(result);
    }
};

namespace {

TEST(CloudMergeTicketTest, PreservesFirstCompletionResult) {
    CloudMergeTicket ticket(7);
    CloudMergeResult first;
    first.sequence = 7;
    first.outcome = CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR;
    first.detail = "first";

    EXPECT_TRUE(CloudMergeTicketTestAccess::Complete(ticket, first));

    CloudMergeResult second;
    second.sequence = 7;
    second.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    second.detail = "second";

    EXPECT_FALSE(CloudMergeTicketTestAccess::Complete(ticket, second));

    CloudMergeResult observed;
    EXPECT_TRUE(CloudMergeTicketTestAccess::Wait(ticket, observed));
    EXPECT_EQ(observed.outcome, CloudMergeOutcome::MERGE_COMPLETED_LAND_AIR);
    EXPECT_EQ(observed.detail, "first");
}

TEST(CloudMergeTicketTest, WakesWaitingThread) {
    CloudMergeTicket ticket(8);
    CloudMergeResult observed;
    std::thread waiter([&ticket, &observed]() {
        EXPECT_TRUE(CloudMergeTicketTestAccess::Wait(ticket, observed));
    });

    CloudMergeResult result;
    result.sequence = 8;
    result.outcome = CloudMergeOutcome::CANCELLED_SHUTDOWN;
    result.detail = "shutdown";
    EXPECT_TRUE(CloudMergeTicketTestAccess::Complete(ticket, result));
    waiter.join();

    EXPECT_EQ(observed.outcome, CloudMergeOutcome::CANCELLED_SHUTDOWN);
}

}  // namespace
}  // namespace ORB_SLAM3
