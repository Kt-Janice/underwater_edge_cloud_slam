// Visual-only diagnostic only; this is outside the formal SVIn2/VIO runtime.
// Remove this file, its launch file, and the marked CMake block to remove the feature.
// Diagnostic currently validated only for the non-ARM/x86_64 Frontend detector path.

#include <brisk/brisk-descriptor-extractor.h>
#include <brisk/harris-score-calculator.h>
#include <brisk/scale-space-feature-detector.h>
#include <cv_bridge/cv_bridge.h>
#include <okvis/MultiFrame.hpp>
#include <okvis/RosParametersReader.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kMinimumBearingZ = 1.0e-9;
constexpr size_t kMinimumGeometryMatches = 8;
constexpr int kCameraIndex = 0;
// Matches the current okvis::Frontend constructor default.
// Diagnostic-only copy; keep synchronized only if that default changes.
constexpr double kBriskDetectionAbsoluteThreshold = 800.0;
// Matches the current okvis::Frontend constructor default.
// Diagnostic-only copy; keep synchronized only if that default changes.
constexpr bool kBriskDescriptionRotationInvariance = true;
// Matches the current okvis::Frontend constructor default.
// Diagnostic-only copy; keep synchronized only if that default changes.
constexpr bool kBriskDescriptionScaleInvariance = false;
// Current okvis::Frontend default matching threshold.
// Used only by this standalone diagnostic.
constexpr double kBriskMatchingThreshold = 60.0;

template <typename Distortion>
bool ReadPinholeFocals(
    const std::shared_ptr<const okvis::cameras::CameraBase>& geometry,
    double& fu,
    double& fv) {
  const auto camera = std::dynamic_pointer_cast<
      const okvis::cameras::PinholeCamera<Distortion>>(geometry);
  if (!camera) {
    return false;
  }
  fu = camera->focalLengthU();
  fv = camera->focalLengthV();
  return true;
}

bool NormalizeBearing(const Eigen::Vector3d& bearing, cv::Point2d& point) {
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(bearing[axis])) {
      return false;
    }
  }
  if (std::abs(bearing.z()) <= kMinimumBearingZ) {
    return false;
  }
  point.x = bearing.x() / bearing.z();
  point.y = bearing.y() / bearing.z();
  return std::isfinite(point.x) && std::isfinite(point.y);
}

struct FrameData {
  std::shared_ptr<okvis::MultiFrame> frame;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

struct Metrics {
  size_t keypoints = 0;
  size_t rawMatches = 0;
  size_t goodMatches = 0;
  size_t geometryMatches = 0;
  bool ransacValid = false;
  size_t ransacInliers = 0;
  double inlierRatio = 0.0;
  double meanParallax = 0.0;
  double medianParallax = 0.0;
};

class VisualOnlyDiagnostic {
 public:
  explicit VisualOnlyDiagnostic(ros::NodeHandle& privateNode)
      : privateNode_(privateNode) {
    std::string settingPath;
    std::string imageTopic;
    std::string outputCsv;
    privateNode_.param<std::string>("setting_path", settingPath, "");
    privateNode_.param<std::string>("image_topic", imageTopic, "/davis_left/image_raw");
    privateNode_.param<std::string>("output_csv", outputCsv, "/tmp/svin_visual_diagnostic.csv");
    if (settingPath.empty()) {
      throw std::runtime_error("~setting_path must name a complete OKVIS configuration");
    }

    okvis::RosParametersReader reader(settingPath);
    if (!reader.getParameters(parameters_)) {
      throw std::runtime_error("Unable to read OKVIS parameters from " + settingPath);
    }
    const size_t cameraCount = parameters_.nCameraSystem.numCameras();
    if (cameraCount < 1) {
      throw std::runtime_error("Configuration has no cameras");
    }
    if (cameraCount > 1) {
      ROS_INFO_STREAM("Visual diagnostic uses camera 0 and ignores "
                      << (cameraCount - 1) << " additional camera(s)");
    }
    geometry_ = parameters_.nCameraSystem.cameraGeometry(kCameraIndex);
    if (!geometry_) {
      throw std::runtime_error("Camera 0 has no geometry");
    }
    const double resizeFactor = parameters_.miscParams.resizeFactor;
    if (!std::isfinite(resizeFactor) || resizeFactor <= 0.0) {
      throw std::runtime_error("resizeFactor must be finite and positive");
    }

    // Match the non-ARM detector and extractor constructors in Frontend.cpp.
    detector_.reset(new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
        parameters_.optimization.detectionThreshold,
        parameters_.optimization.detectionOctaves,
        kBriskDetectionAbsoluteThreshold,
        parameters_.optimization.maxNoKeypoints));
    extractor_.reset(new brisk::BriskDescriptorExtractor(
        kBriskDescriptionRotationInvariance,
        kBriskDescriptionScaleInvariance));
    descriptorSize_ = extractor_->descriptorSize();
    if (descriptorSize_ <= 0 || extractor_->descriptorType() != CV_8U) {
      ROS_ERROR("BRISK extractor has invalid descriptor size/type: size=%d type=%d",
                descriptorSize_, extractor_->descriptorType());
      ros::shutdown();
      throw std::runtime_error("Unsupported BRISK descriptor layout");
    }

    switch (parameters_.nCameraSystem.distortionType(kCameraIndex)) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        if (!ReadPinholeFocals<okvis::cameras::RadialTangentialDistortion>(geometry_, fu_, fv_)) {
          FailUnsupportedCamera();
        }
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        if (!ReadPinholeFocals<okvis::cameras::EquidistantDistortion>(geometry_, fu_, fv_)) {
          FailUnsupportedCamera();
        }
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        if (!ReadPinholeFocals<okvis::cameras::RadialTangentialDistortion8>(geometry_, fu_, fv_)) {
          FailUnsupportedCamera();
        }
        break;
      }
      default: {
        FailUnsupportedCamera();
      }
    }
    if (!std::isfinite(fu_) || !std::isfinite(fv_) || fu_ <= 0.0 || fv_ <= 0.0) {
      ROS_ERROR("Camera 0 has invalid focal lengths: fu=%g fv=%g", fu_, fv_);
      ros::shutdown();
      throw std::runtime_error("Invalid camera 0 focal lengths");
    }
    ransacThreshold_ = 1.0 / ((fu_ + fv_) / 2.0);

    if (parameters_.histogramParams.histogramMethod == okvis::CLAHE) {
      clahe_ = cv::createCLAHE();
      clahe_->setClipLimit(parameters_.histogramParams.claheClipLimit);
      clahe_->setTilesGridSize(cv::Size(parameters_.histogramParams.claheTilesGridSize,
                                        parameters_.histogramParams.claheTilesGridSize));
    }

    csv_.open(outputCsv.c_str(), std::ios::out | std::ios::trunc);
    if (!csv_) {
      throw std::runtime_error("Cannot open CSV: " + outputCsv);
    }
    csv_ << "timestamp,frame_id,num_keypoints,num_raw_matches,num_good_matches,"
            "num_geometry_matches,ransac_valid,num_ransac_inliers,inlier_ratio,"
            "mean_parallax_px,median_parallax_px\n";
    csv_.flush();

    ROS_WARN("No IMU gravity estimate is available; visual diagnostic uses a fixed "
             "extraction direction equivalent to identity T_WC. This does not "
             "reproduce the real VIO orientation condition");
    ROS_INFO_STREAM("Visual diagnostic image topic=" << imageTopic
                    << " calibration=" << settingPath << " CSV=" << outputCsv
                    << " descriptor_bytes=" << descriptorSize_);
    imageSubscriber_ = privateNode_.subscribe(imageTopic, 10,
                                             &VisualOnlyDiagnostic::ImageCallback, this);
  }

  ~VisualOnlyDiagnostic() {
    if (csv_.is_open()) {
      csv_.flush();
      csv_.close();
    }
    ROS_INFO("Visual diagnostic received=%llu written=%llu skipped=%llu",
             static_cast<unsigned long long>(received_),
             static_cast<unsigned long long>(written_),
             static_cast<unsigned long long>(skipped_));
  }

 private:
  void FailUnsupportedCamera() {
    ROS_ERROR("Unsupported camera 0 model: distortion_type=%d, geometry=%s",
              static_cast<int>(parameters_.nCameraSystem.distortionType(kCameraIndex)),
              typeid(*geometry_).name());
    ros::shutdown();
    throw std::runtime_error("Unsupported camera 0 model");
  }

  bool Preprocess(const sensor_msgs::ImageConstPtr& message, cv::Mat& raw, cv::Mat& processed) {
    try {
      const cv_bridge::CvImageConstPtr converted = cv_bridge::toCvCopy(message);
      if (message->encoding == sensor_msgs::image_encodings::MONO8) {
        raw = converted->image;
      } else if (message->encoding == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(converted->image, raw, cv::COLOR_BGR2GRAY);
      } else if (message->encoding == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(converted->image, raw, cv::COLOR_RGB2GRAY);
      } else {
        ROS_WARN_STREAM("Skipping unsupported image encoding: " << message->encoding);
        return false;
      }
      if (raw.empty()) {
        ROS_WARN("Skipping empty image");
        return false;
      }

      cv::Mat resized;
      const double factor = parameters_.miscParams.resizeFactor;
      if (factor != 1.0) {
        cv::resize(raw, resized, cv::Size(), factor, factor);
      } else {
        resized = raw.clone();
      }
      cv::Mat filtered;
      if (parameters_.optimization.useMedianFilter) {
        cv::medianBlur(resized, filtered, 3);
      } else {
        filtered = resized.clone();
      }
      if (parameters_.histogramParams.histogramMethod == okvis::CLAHE) {
        clahe_->apply(filtered, processed);
      } else if (parameters_.histogramParams.histogramMethod == okvis::HISTOGRAM) {
        cv::equalizeHist(filtered, processed);
      } else {
        processed = filtered;
      }
      return true;
    } catch (const cv::Exception& error) {
      ROS_WARN_STREAM("Skipping image after OpenCV error: " << error.what());
      return false;
    } catch (const cv_bridge::Exception& error) {
      ROS_WARN_STREAM("Skipping image after cv_bridge error: " << error.what());
      return false;
    }
  }

  FrameData ExtractFrame(const sensor_msgs::ImageConstPtr& message, const cv::Mat& image) {
    const okvis::Time timestamp(message->header.stamp.sec, message->header.stamp.nsec);
    FrameData current;
    current.frame.reset(new okvis::MultiFrame(parameters_.nCameraSystem, timestamp, nextFrameId_));
    current.frame->setImage(kCameraIndex, image);

    current.frame->setDetector(kCameraIndex, detector_);
    current.frame->setExtractor(kCameraIndex, extractor_);
    current.frame->detect(kCameraIndex);
    // Frontend.cpp computes T_WC.inverse().C() * (0, 0, -1). The diagnostic
    // identity T_WC assumption makes this fixed direction (0, 0, -1).
    const Eigen::Vector3d extractionDirection(0.0, 0.0, -1.0);
    current.frame->describe(kCameraIndex, extractionDirection);

    const size_t count = current.frame->numKeypoints(kCameraIndex);
    current.keypoints.reserve(count);
    if (count == 0) {
      return current;
    }
    current.descriptors.create(static_cast<int>(count), descriptorSize_, CV_8U);
    for (size_t index = 0; index < count; ++index) {
      cv::KeyPoint keypoint;
      current.frame->getCvKeypoint(kCameraIndex, index, keypoint);
      current.keypoints.push_back(keypoint);
      const unsigned char* descriptor = current.frame->keypointDescriptor(kCameraIndex, index);
      if (descriptor == nullptr) {
        throw std::runtime_error("Missing BRISK descriptor for detected keypoint");
      }
      std::memcpy(current.descriptors.ptr(static_cast<int>(index)), descriptor,
                  static_cast<size_t>(descriptorSize_));
    }
    return current;
  }

  Metrics Measure(const FrameData& current) const {
    Metrics metrics;
    metrics.keypoints = current.keypoints.size();
    if (!hasPrevious_ || previous_.descriptors.empty() || current.descriptors.empty()) {
      return metrics;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> forward;
    std::vector<cv::DMatch> backward;
    matcher.match(previous_.descriptors, current.descriptors, forward);
    matcher.match(current.descriptors, previous_.descriptors, backward);
    metrics.rawMatches = forward.size();

    std::vector<int> reverseNearest(static_cast<size_t>(current.descriptors.rows), -1);
    for (const cv::DMatch& match : backward) {
      if (match.queryIdx >= 0 && match.queryIdx < current.descriptors.rows) {
        reverseNearest[static_cast<size_t>(match.queryIdx)] = match.trainIdx;
      }
    }

    std::vector<double> parallax;
    std::vector<cv::Point2d> normalizedPrevious;
    std::vector<cv::Point2d> normalizedCurrent;
    const double resizeFactor = parameters_.miscParams.resizeFactor;
    for (const cv::DMatch& match : forward) {
      if (match.queryIdx < 0 || match.trainIdx < 0 ||
          match.queryIdx >= static_cast<int>(previous_.keypoints.size()) ||
          match.trainIdx >= static_cast<int>(current.keypoints.size()) ||
          match.distance >= kBriskMatchingThreshold ||
          reverseNearest[static_cast<size_t>(match.trainIdx)] != match.queryIdx) {
        continue;
      }
      ++metrics.goodMatches;
      const cv::Point2f previousPixel = previous_.keypoints[static_cast<size_t>(match.queryIdx)].pt;
      const cv::Point2f currentPixel = current.keypoints[static_cast<size_t>(match.trainIdx)].pt;

      // Descriptor-consistent match displacement only; neither RANSAC-inlier
      // motion nor metric camera translation. Report original-image pixels.
      parallax.push_back(cv::norm(previousPixel - currentPixel) / resizeFactor);

      Eigen::Vector3d previousBearing;
      Eigen::Vector3d currentBearing;
      if (!geometry_->backProject(Eigen::Vector2d(previousPixel.x, previousPixel.y), &previousBearing) ||
          !geometry_->backProject(Eigen::Vector2d(currentPixel.x, currentPixel.y), &currentBearing)) {
        continue;
      }
      cv::Point2d previousNormalized;
      cv::Point2d currentNormalized;
      if (!NormalizeBearing(previousBearing, previousNormalized) ||
          !NormalizeBearing(currentBearing, currentNormalized)) {
        continue;
      }
      normalizedPrevious.push_back(previousNormalized);
      normalizedCurrent.push_back(currentNormalized);
    }
    metrics.geometryMatches = normalizedPrevious.size();
    if (!parallax.empty()) {
      for (double displacement : parallax) {
        metrics.meanParallax += displacement;
      }
      metrics.meanParallax /= static_cast<double>(parallax.size());
      std::sort(parallax.begin(), parallax.end());
      const size_t middle = parallax.size() / 2;
      if (parallax.size() % 2 == 0) {
        metrics.medianParallax = (parallax[middle - 1] + parallax[middle]) / 2.0;
      } else {
        metrics.medianParallax = parallax[middle];
      }
    }

    // Eight correspondences/inliers are a diagnostic stability rule; OpenCV's
    // five-point solver has a lower theoretical minimum.
    if (metrics.geometryMatches < kMinimumGeometryMatches) {
      return metrics;
    }
    try {
      cv::Mat mask;
      const cv::Mat essential = cv::findEssentialMat(
          normalizedPrevious, normalizedCurrent, cv::Mat::eye(3, 3, CV_64F),
          cv::RANSAC, 0.999, ransacThreshold_, mask);
      if (essential.empty() || essential.cols != 3 || essential.rows < 3 ||
          essential.rows % 3 != 0 || !cv::checkRange(essential) ||
          mask.empty() || mask.type() != CV_8UC1 ||
          mask.total() != metrics.geometryMatches) {
        return metrics;
      }
      const int inliers = cv::countNonZero(mask);
      if (inliers < static_cast<int>(kMinimumGeometryMatches)) {
        return metrics;
      }
      metrics.ransacValid = true;
      metrics.ransacInliers = static_cast<size_t>(inliers);
      metrics.inlierRatio = static_cast<double>(inliers) /
                            static_cast<double>(metrics.geometryMatches);
    } catch (const cv::Exception& error) {
      ROS_WARN_STREAM("Essential-RANSAC produced no usable result: " << error.what());
    }
    return metrics;
  }

  void WriteCsvRow(const ros::Time& stamp, uint64_t frameId, const Metrics& metrics) {
    csv_ << stamp.sec << '.' << std::setw(9) << std::setfill('0') << stamp.nsec
         << std::setfill(' ') << ',' << frameId << ',' << metrics.keypoints << ','
         << metrics.rawMatches << ',' << metrics.goodMatches << ','
         << metrics.geometryMatches << ',' << static_cast<int>(metrics.ransacValid)
         << ',' << metrics.ransacInliers << ',' << metrics.inlierRatio << ','
         << metrics.meanParallax << ',' << metrics.medianParallax << '\n';
    csv_.flush();
    if (!csv_) {
      throw std::runtime_error("Failed to write visual diagnostic CSV");
    }
  }

  void ImageCallback(const sensor_msgs::ImageConstPtr& message) {
    ++received_;
    if (hasPrevious_ && message->header.stamp <= previousStamp_) {
      ++skipped_;
      ROS_WARN_STREAM("Skipping repeated/out-of-order image timestamp " << message->header.stamp);
      return;
    }
    cv::Mat raw;
    cv::Mat processed;
    if (!Preprocess(message, raw, processed)) {
      ++skipped_;
      return;
    }
    const int expectedWidth = static_cast<int>(geometry_->imageWidth());
    const int expectedHeight = static_cast<int>(geometry_->imageHeight());
    if (processed.cols != expectedWidth || processed.rows != expectedHeight) {
      ++skipped_;
      ROS_ERROR("Camera 0 image/calibration size mismatch: raw=%dx%d processed=%dx%d calibration=%dx%d",
                raw.cols, raw.rows, processed.cols, processed.rows, expectedWidth, expectedHeight);
      ros::shutdown();
      return;
    }

    try {
      FrameData current = ExtractFrame(message, processed);
      const Metrics metrics = Measure(current);
      WriteCsvRow(message->header.stamp, nextFrameId_, metrics);
      previous_ = std::move(current);
      previousStamp_ = message->header.stamp;
      hasPrevious_ = true;
      ++nextFrameId_;
      ++written_;
    } catch (const std::exception& error) {
      ++skipped_;
      ROS_ERROR_STREAM("Visual diagnostic processing failed: " << error.what());
      ros::shutdown();
    }
  }

  ros::NodeHandle privateNode_;
  ros::Subscriber imageSubscriber_;
  okvis::VioParameters parameters_;
  std::shared_ptr<cv::FeatureDetector> detector_;
  std::shared_ptr<brisk::BriskDescriptorExtractor> extractor_;
  std::shared_ptr<const okvis::cameras::CameraBase> geometry_;
  cv::Ptr<cv::CLAHE> clahe_;
  std::ofstream csv_;
  FrameData previous_;
  ros::Time previousStamp_;
  bool hasPrevious_ = false;
  uint64_t nextFrameId_ = 0;
  uint64_t received_ = 0;
  uint64_t skipped_ = 0;
  uint64_t written_ = 0;
  int descriptorSize_ = 0;
  double fu_ = 0.0;
  double fv_ = 0.0;
  double ransacThreshold_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "svin_visual_diagnostic");
  ros::NodeHandle privateNode("~");
  try {
    VisualOnlyDiagnostic diagnostic(privateNode);
    ros::spin();
    return 0;
  } catch (const std::exception& error) {
    ROS_ERROR_STREAM("Visual diagnostic could not start: " << error.what());
    ros::shutdown();
    return 1;
  }
}
