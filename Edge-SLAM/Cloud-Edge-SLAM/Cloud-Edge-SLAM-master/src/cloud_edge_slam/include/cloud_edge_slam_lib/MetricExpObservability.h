#ifndef METRIC_EXP_OBSERVABILITY_H
#define METRIC_EXP_OBSERVABILITY_H

#include <Eigen/Core>

namespace ORB_SLAM3
{

struct MetricExpHessianSummary
{
    MetricExpHessianSummary();

    bool valid;
    long rank;
    bool rankDeficient;
    bool negativeEigenvalue;
    double rawMinEigenvalue;
    Eigen::Matrix<double, 6, 1> eigenvalues;
    double trace;
    double minTraceRatio;
    double minMaxRatio;
    double log10Condition;
};

struct MetricExpTransObservabilitySummary
{
    MetricExpTransObservabilitySummary();

    bool valid;
    bool negativeEigenvalue;
    double rawMinEigenvalue;
    Eigen::Matrix3d matrix;
    Eigen::Matrix<double, 3, 1> eigenvalues;
    double trace;
    double minTraceRatio;
    double minMaxRatio;
    double log10Condition;
};

MetricExpHessianSummary SummarizeMetricExpHessian(
    const Eigen::Matrix<double, 6, 6>& hessian,
    long edgeCount);

MetricExpTransObservabilitySummary SummarizeMetricExpTransObservability(
    const Eigen::Matrix<double, 6, 6>& hessian);

} // namespace ORB_SLAM3

#endif // METRIC_EXP_OBSERVABILITY_H
