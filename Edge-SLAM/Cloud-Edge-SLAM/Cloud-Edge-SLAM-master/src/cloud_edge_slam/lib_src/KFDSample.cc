#include "KFDSample.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

#include <cmath>
#include <limits>

KFDSample::KFDSample(int nfeatures, int nlevels, int iniThFAST, int minThFAST, float scaleFactor, Mat &Frame, float TimeStamp) {
    // Initialize
    this->nfeatures = nfeatures;
    this->minThFAST = minThFAST;
    this->nlevels = nlevels;
    this->iniThFAST = iniThFAST;
    this->scaleFactor = scaleFactor;
    this->InitORBextractor(this->nfeatures, this->scaleFactor, this->nlevels, this->iniThFAST, this->minThFAST);
    // this->SetInitialFrame(Frame);
    // this->frame1 = Frame;
    this->ltframe = TimeStamp;
}

KFDSample::KFDSample() {
    InitORBextractor(this->nfeatures, this->scaleFactor, this->nlevels, this->iniThFAST, this->minThFAST);
    InitPDKFselector();
}

// Init PD keyframe select controller
void KFDSample::InitPDKFselector(float Kp, float Kd, float th) {
    this->mpPDcontroller = new PD(Kp, Kd);
    this->mpPDcontroller->setSetpoint(th);
    Printinfo(Kp, Kd, th);
}

void KFDSample::InitPDKFselector() {
    this->mpPDcontroller = new PD(this->Kp, this->Kd);
    this->mpPDcontroller->setSetpoint(this->th);
}

void KFDSample::SetPDKFselectorParams(float Kp_, float Kd_, float th_) {
    this->Kp = Kp_;
    this->Kd = Kd_;
    this->th = th_;
    this->mpPDcontroller->setKp(this->Kp);
    this->mpPDcontroller->setKd(this->Kd);
    this->mpPDcontroller->setSetpoint(this->th);
}

Mat KFDSample::GetKF() {
    Mat KF = KFset.back();
    return KF;
}
vector<Mat> KFDSample::GetAllKF() {
    vector<Mat> allKF = KFset;
    return allKF;
}

// set threshold
void KFDSample::SetThreshold(const float TH) {
    this->th = TH;
}

void KFDSample::SetDisplay() {
    this->show = true;
}

// Input Frame
void KFDSample::SetNextFrame(const Mat &InputArray) {
    this->frame2 = InputArray.clone();
    this->imnext = InputArray.clone();
    // cvtColor(this->frame2, this->imnext, COLOR_BGR2GRAY);
    // imshow("grayscale", this->imnext);
    // waitKey(20);
}

// Select Good points After tracking
void KFDSample::SelectGoodPts() {
    for (uint i = 0; i < this->old.size(); i++) {
        if (this->status[i] == 1) {
            this->good_next.push_back(this->next[i]);
            this->good_old.push_back(this->old[i]);
        }
    }
}

void KFDSample::Reset() {
    this->old.clear();
}

bool KFDSample::Step(const Mat &inputIm, double timeStamp) {
    bool result = false;

    // // 等差采样
    // static int nFlag = 0;
    // nFlag++;
    // // if (nFlag == 3) {
    // if (nFlag == 5) {
    //     result = true;
    //     nFlag = 0;
    // }

    // 光流场景
    cv::Mat im = inputIm.clone();
    if (im.type() == CV_8UC3) {
        cvtColor(im, im, COLOR_BGR2GRAY);
    } else if (im.type() == CV_8UC1) {
    } else {
        cerr << "error image type" << endl;
    }

    // @note initial
    if (this->old.empty()) {
        this->ltframe = timeStamp;
        this->frame1 = im.clone();
        this->imprvs = im.clone();
        this->mpORBextractor->operator()(this->imprvs, cv::Mat(), this->mvKeys, this->mDescriptors, vLapping);
        KeyPoint::convert(mvKeys, this->old);
        KFset.push_back(frame1);
        if (this->show) {
            Mat nkeypointstemp;
            drawKeypoints(imprvs, mvKeys, nkeypointstemp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            imshow("ORB Feature", nkeypointstemp);
            waitKey(10);
        }
        result = true;
        return result;
    }

    // @note step
    this->SetNextFrame(im);
    // imshow("frame1", this->imprvs);
    // imshow("frame2", this->imnext);
    // waitKey(10);
    calcOpticalFlowPyrLK(this->imprvs, this->imnext, this->old,
                         this->next, this->status, this->err, Size(31, 31), 2, this->criteria);
    this->SelectGoodPts();
    float nkpt = good_next.size();

    // @note print
    // cout.precision(16);
    // cout << "At Timestamp: " << timeStamp << endl;
    // cout.precision(5);
    // cout << "Num of Good Points:" << nkpt << endl;

    this->moptf = Calmoptflmag(this->good_old, this->good_next, nkpt);
    float vmOptflow = this->moptf;
    float vPDKFth = mpPDcontroller->update(vmOptflow, timeStamp - this->ltframe);
    float TH = vmOptflow + vPDKFth;
    // @note print
    // cout << "PD Output: " << vPDKFth << endl;
    // cout << "Select Threshold: " << TH << endl;
    // select
    if (this->moptf > TH) {
        // @note print
        // cout << "New Keyframe Selected" << endl;
        this->mpORBextractor->operator()(this->imnext, cv::Mat(), this->mvKeys, this->mDescriptors, vLapping);
        KeyPoint::convert(mvKeys, this->old);
        KFset.push_back(frame2);
        if (this->show) {
            vector<KeyPoint> keys;
            Mat keypointstemp;
            KeyPoint::convert(good_next, keys);
            drawKeypoints(imnext, keys, keypointstemp, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            imshow("ORB Keypoints", keypointstemp);
            waitKey(10);
        }
        result = true;
    } else {
        this->old = this->next;
    }
    this->ltframe = timeStamp;
    this->imprvs = this->imnext.clone();
    // this->old = this->next;
    this->good_next.clear();
    this->good_old.clear();

    return result;
}

bool KFDSample::IsValidLandAirImage(const Mat &image) const {
    if (image.empty()) {
        return false;
    }

    if (image.type() == CV_8UC1 || image.type() == CV_8UC3) {
        return true;
    }

    return false;
}

bool KFDSample::AreFiniteLandAirPoints(const vector<Point2f> &points) const {
    for (const Point2f &point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }
    return true;
}

bool KFDSample::BuildLandAirBaseline(const Mat &image, double timeStamp) {
    if (!IsValidLandAirImage(image) || !std::isfinite(timeStamp)) {
        return false;
    }

    Mat gray;
    if (image.type() == CV_8UC3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    if (gray.empty() || gray.type() != CV_8UC1) {
        return false;
    }

    ltframe = timeStamp;
    frame1 = gray.clone();
    imprvs = gray.clone();
    frame2.release();
    imnext.release();
    next.clear();
    status.clear();
    err.clear();
    good_old.clear();
    good_next.clear();
    try {
        mpORBextractor->operator()(imprvs, cv::Mat(), mvKeys, mDescriptors, vLapping);
        KeyPoint::convert(mvKeys, old);
    } catch (const std::exception &) {
        ResetLandAirEpisode();
        return false;
    }
    if (old.empty()) {
        ResetLandAirEpisode();
        return false;
    }
    KFset.push_back(frame1);

    if (show) {
        Mat keypoints;
        drawKeypoints(imprvs, mvKeys, keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("ORB Feature", keypoints);
        waitKey(10);
    }

    return true;
}

void KFDSample::ResetLandAirEpisode() {
    old.clear();
    next.clear();
    good_old.clear();
    good_next.clear();
    status.clear();
    err.clear();
    mvKeys.clear();
    mDescriptors.release();
    frame1.release();
    frame2.release();
    imprvs.release();
    imnext.release();
    ltframe = std::numeric_limits<double>::quiet_NaN();
    moptf = 0.0F;

    if (mpPDcontroller != nullptr) {
        mpPDcontroller->Reset();
    }
}

bool KFDSample::StepForLandAir(const Mat &inputIm, double timeStamp, bool *pAccepted) {
    if (pAccepted != nullptr) {
        *pAccepted = false;
    }

    ++mnLandAirStepCallCount;

    if (!IsValidLandAirImage(inputIm) || !std::isfinite(timeStamp)) {
        ResetLandAirEpisode();
        return false;
    }

    Mat image;
    if (inputIm.type() == CV_8UC3) {
        cvtColor(inputIm, image, COLOR_BGR2GRAY);
    } else {
        image = inputIm.clone();
    }

    if (image.empty() || image.type() != CV_8UC1) {
        ResetLandAirEpisode();
        return false;
    }

    if (old.empty()) {
        const bool selected = BuildLandAirBaseline(image, timeStamp);
        if (selected && pAccepted != nullptr) {
            *pAccepted = true;
        }
        return selected;
    }

    if (imprvs.empty() || imprvs.size() != image.size() || imprvs.type() != image.type() ||
        !std::isfinite(ltframe) || timeStamp <= ltframe || !AreFiniteLandAirPoints(old)) {
        ResetLandAirEpisode();
        return false;
    }

    SetNextFrame(image);
    if (imnext.empty() || imnext.size() != imprvs.size() || imnext.type() != imprvs.type()) {
        ResetLandAirEpisode();
        return false;
    }

    try {
        ++mnLandAirLkCallCount;
        calcOpticalFlowPyrLK(imprvs, imnext, old, next, status, err, Size(31, 31), 2, criteria);
    } catch (const cv::Exception &) {
        ResetLandAirEpisode();
        return false;
    }

    if (next.size() != old.size() || status.size() != old.size() || err.size() != old.size() ||
        !AreFiniteLandAirPoints(next)) {
        ResetLandAirEpisode();
        return false;
    }

    good_old.clear();
    good_next.clear();
    SelectGoodPts();
    if (good_old.empty() || good_next.empty() || good_old.size() != good_next.size()) {
        ResetLandAirEpisode();
        return false;
    }

    const float pointCount = static_cast<float>(good_next.size());
    moptf = Calmoptflmag(good_old, good_next, pointCount);
    const float controllerOutput = mpPDcontroller->update(moptf, timeStamp - ltframe);
    const float selectionThreshold = moptf + controllerOutput;
    bool selected = false;

    if (moptf > selectionThreshold) {
        try {
            mpORBextractor->operator()(imnext, cv::Mat(), mvKeys, mDescriptors, vLapping);
            KeyPoint::convert(mvKeys, old);
        } catch (const std::exception &) {
            ResetLandAirEpisode();
            return false;
        }
        if (old.empty()) {
            ResetLandAirEpisode();
            return false;
        }
        KFset.push_back(frame2);
        selected = true;
        if (show) {
            vector<KeyPoint> keys;
            Mat keypoints;
            KeyPoint::convert(good_next, keys);
            drawKeypoints(imnext, keys, keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            imshow("ORB Keypoints", keypoints);
            waitKey(10);
        }
    } else {
        old = next;
    }

    ltframe = timeStamp;
    imprvs = imnext.clone();
    good_next.clear();
    good_old.clear();
    if (pAccepted != nullptr) {
        *pAccepted = true;
    }

    return selected;
}

std::uint64_t KFDSample::GetLandAirStepCallCount() const {
    return mnLandAirStepCallCount;
}

std::uint64_t KFDSample::GetLandAirLkCallCount() const {
    return mnLandAirLkCallCount;
}

//print PD
void Printinfo(float Kp, float Kd, float setpoint) {
    cout << endl
         << "KeyFrame PD Selector Parameters: " << endl;
    cout << "- Kp: " << Kp << endl;
    cout << "- Kd: " << Kd << endl;
    cout << "- Setpoint" << setpoint << endl;
}

float Calmoptflmag(const vector<Point2f> prvs, const vector<Point2f> next, const int nkpt) {
    float sum = 0;
    float moptf = 0;
    for (int i = 0; i < nkpt; i++) {
        float a = sqrt((next[i].x - prvs[i].x) * (next[i].x - prvs[i].x) + (next[i].y - prvs[i].y) * (next[i].y - prvs[i].y));
        sum += a;
    }
    moptf = sum / nkpt;
    // @note print
    // cout.precision(4);
    // cout << "Magnitude of Optical flow:" << moptf << endl;
    return moptf;
}
