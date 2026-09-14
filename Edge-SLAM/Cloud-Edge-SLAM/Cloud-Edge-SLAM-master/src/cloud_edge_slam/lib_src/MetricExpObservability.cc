#include "MetricExpObservability.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <limits>

namespace ORB_SLAM3
{
namespace
{
const double kAbsoluteNumericEpsilon = 1e-15;
const double kRelativeEigenTolerance = 1e-12;

double QuietNaN()
{
    return std::numeric_limits<double>::quiet_NaN();
}

bool IsFinite(const Eigen::MatrixXd& matrix)
{
    return matrix.array().isFinite().all();
}

struct EigenvalueSummary
{
    EigenvalueSummary()
        : valid(false),
          rank(0),
          rankDeficient(true),
          negativeEigenvalue(false),
          rawMinEigenvalue(QuietNaN()),
          trace(QuietNaN()),
          minTraceRatio(QuietNaN()),
          minMaxRatio(QuietNaN()),
          log10Condition(QuietNaN())
    {
    }

    bool valid;
    long rank;
    bool rankDeficient;
    bool negativeEigenvalue;
    double rawMinEigenvalue;
    Eigen::VectorXd eigenvalues;
    double trace;
    double minTraceRatio;
    double minMaxRatio;
    double log10Condition;
};

EigenvalueSummary SummarizeSymmetricMatrix(const Eigen::MatrixXd& input)
{
    EigenvalueSummary summary;
    if (input.rows() == 0 || input.rows() != input.cols() || !IsFinite(input))
    {
        return summary;
    }

    const Eigen::MatrixXd symmetric = (input + input.transpose()) * 0.5;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
    if (solver.info() != Eigen::Success)
    {
        return summary;
    }

    const Eigen::VectorXd rawEigenvalues = solver.eigenvalues();
    if (rawEigenvalues.size() == 0 || !rawEigenvalues.array().isFinite().all())
    {
        return summary;
    }

    summary.rawMinEigenvalue = rawEigenvalues(0);
    const double lambdaMax = rawEigenvalues(rawEigenvalues.size() - 1);
    const double positiveLambdaMax = std::max(lambdaMax, 0.0);
    const double eigEpsilon = std::max(positiveLambdaMax * kRelativeEigenTolerance,
                                       kAbsoluteNumericEpsilon);
    const double negativeTolerance = std::max(positiveLambdaMax * kRelativeEigenTolerance,
                                               kAbsoluteNumericEpsilon);

    summary.eigenvalues = rawEigenvalues;
    for (Eigen::Index index = 0; index < summary.eigenvalues.size(); ++index)
    {
        if (summary.eigenvalues(index) < -negativeTolerance)
        {
            summary.negativeEigenvalue = true;
            return summary;
        }
        if (summary.eigenvalues(index) < 0.0)
        {
            summary.eigenvalues(index) = 0.0;
        }
    }

    summary.valid = true;
    summary.trace = summary.eigenvalues.sum();
    const double clampedLambdaMin = summary.eigenvalues(0);
    const double clampedLambdaMax = summary.eigenvalues(summary.eigenvalues.size() - 1);
    for (Eigen::Index index = 0; index < summary.eigenvalues.size(); ++index)
    {
        if (summary.eigenvalues(index) > eigEpsilon)
        {
            ++summary.rank;
        }
    }
    summary.rankDeficient = summary.rank < summary.eigenvalues.size();

    if (summary.trace > 0.0 && clampedLambdaMax > 0.0)
    {
        if (summary.rankDeficient)
        {
            summary.minTraceRatio = 0.0;
            summary.minMaxRatio = 0.0;
        }
        else
        {
            summary.minTraceRatio = clampedLambdaMin / summary.trace;
            summary.minMaxRatio = clampedLambdaMin / clampedLambdaMax;
        }
    }

    const double lambdaEffective = std::max(clampedLambdaMin, eigEpsilon);
    const double conditionNumerator = std::max(clampedLambdaMax, eigEpsilon);
    summary.log10Condition = std::log10(conditionNumerator / lambdaEffective);
    return summary;
}

template<int Size>
void CopyEigenvalueSummary(const EigenvalueSummary& source,
                           Eigen::Matrix<double, Size, 1>* destination)
{
    destination->setConstant(QuietNaN());
    if (source.eigenvalues.size() == Size)
    {
        *destination = source.eigenvalues;
    }
}

} // namespace

MetricExpHessianSummary::MetricExpHessianSummary()
    : valid(false),
      rank(0),
      rankDeficient(true),
      negativeEigenvalue(false),
      rawMinEigenvalue(QuietNaN()),
      trace(QuietNaN()),
      minTraceRatio(QuietNaN()),
      minMaxRatio(QuietNaN()),
      log10Condition(QuietNaN())
{
    eigenvalues.setConstant(QuietNaN());
}

MetricExpTransObservabilitySummary::MetricExpTransObservabilitySummary()
    : valid(false),
      negativeEigenvalue(false),
      rawMinEigenvalue(QuietNaN()),
      trace(QuietNaN()),
      minTraceRatio(QuietNaN()),
      minMaxRatio(QuietNaN()),
      log10Condition(QuietNaN())
{
    matrix.setConstant(QuietNaN());
    eigenvalues.setConstant(QuietNaN());
}

MetricExpHessianSummary SummarizeMetricExpHessian(
    const Eigen::Matrix<double, 6, 6>& hessian,
    long edgeCount)
{
    MetricExpHessianSummary result;
    if (edgeCount <= 0)
    {
        return result;
    }

    const EigenvalueSummary summary = SummarizeSymmetricMatrix(hessian);
    result.valid = summary.valid;
    result.rank = summary.rank;
    result.rankDeficient = summary.rankDeficient;
    result.negativeEigenvalue = summary.negativeEigenvalue;
    result.rawMinEigenvalue = summary.rawMinEigenvalue;
    CopyEigenvalueSummary<6>(summary, &result.eigenvalues);
    result.trace = summary.trace;
    result.minTraceRatio = summary.minTraceRatio;
    result.minMaxRatio = summary.minMaxRatio;
    result.log10Condition = summary.log10Condition;
    return result;
}

MetricExpTransObservabilitySummary SummarizeMetricExpTransObservability(
    const Eigen::Matrix<double, 6, 6>& hessian)
{
    MetricExpTransObservabilitySummary result;
    if (!hessian.array().isFinite().all())
    {
        return result;
    }

    const Eigen::Matrix3d hrr = (hessian.block<3, 3>(0, 0) +
                                  hessian.block<3, 3>(0, 0).transpose()) * 0.5;
    const EigenvalueSummary rotationSummary = SummarizeSymmetricMatrix(hrr);
    if (!rotationSummary.valid || rotationSummary.rank < 3)
    {
        return result;
    }

    Eigen::LDLT<Eigen::Matrix3d> ldlt(hrr);
    if (ldlt.info() != Eigen::Success || !ldlt.vectorD().array().isFinite().all())
    {
        return result;
    }

    const Eigen::Matrix3d hrt = hessian.block<3, 3>(0, 3);
    const Eigen::Matrix3d htr = hessian.block<3, 3>(3, 0);
    const Eigen::Matrix3d htt = hessian.block<3, 3>(3, 3);
    const Eigen::Matrix3d solved = ldlt.solve(hrt);
    if (ldlt.info() != Eigen::Success || !solved.array().isFinite().all())
    {
        return result;
    }

    const Eigen::Matrix3d marginal = htt - htr * solved;
    const Eigen::Matrix3d symmetricMarginal =
        (marginal + marginal.transpose()) * 0.5;
    const EigenvalueSummary summary = SummarizeSymmetricMatrix(marginal);
    result.valid = summary.valid;
    result.negativeEigenvalue = summary.negativeEigenvalue;
    result.rawMinEigenvalue = summary.rawMinEigenvalue;
    if (summary.valid)
    {
        result.matrix = symmetricMarginal;
    }
    CopyEigenvalueSummary<3>(summary, &result.eigenvalues);
    result.trace = summary.trace;
    result.minTraceRatio = summary.minTraceRatio;
    result.minMaxRatio = summary.minMaxRatio;
    result.log10Condition = summary.log10Condition;
    return result;
}

} // namespace ORB_SLAM3
