#include "CloudMerging.h"

#include <iostream>

namespace ORB_SLAM3 {

CloudMergeTicketPtr CloudMerging::InsertCloudMapWithTicket(Map *pMap) {
    static_cast<void>(pMap);
    std::cerr
        << "CloudMerging completion tickets are unavailable in a legacy "
        << "merger target" << std::endl;
    return CloudMergeTicketPtr();
}

bool CloudMerging::WaitForCompletion(
    const CloudMergeTicketPtr &ticket,
    CloudMergeResult &result) {
    result.sequence = 0U;
    if (ticket != nullptr) {
        result.sequence = ticket->GetSequence();
    }
    result.outcome = CloudMergeOutcome::FAILED_EXCEPTION;
    result.detail =
        "CloudMerging completion tickets are unavailable in a legacy "
        "merger target";
    return false;
}

}  // namespace ORB_SLAM3
