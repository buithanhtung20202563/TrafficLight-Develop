#include "Vehicle.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>

Vehicle::Vehicle() : threshold_(0.5f), mtx_() {}

Vehicle::~Vehicle() { Destroy(); }

bool Vehicle::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::mutex> lock(mtx_);
    threshold_ = threshold;
    return detector_.Initialize(modelDir.c_str(), threshold_);
}

bool Vehicle::ConfigureParameters(const std::vector<cv::Point>& detectArea,
                                 const std::vector<cv::Point>& crossingLine,
                                 const std::vector<cv::Point>& directionLine) {
    std::lock_guard<std::mutex> lock(mtx_);
    detectArea_ = detectArea;
    crossingLine_ = crossingLine;
    directionLine_ = directionLine;
    return true;
}

bool Vehicle::Optimize(bool fp16) {
    std::lock_guard<std::mutex> lock(mtx_);
    return detector_.Optimize(fp16);
}

std::vector<ANSCENTER::Object> Vehicle::DetectVehicles(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ANSCENTER::Object> detected;
    detector_.RunInference(input, cameraId.c_str(), detected);
    // TODO: Filter by detectArea_ if needed
    return detected;
}

bool Vehicle::HasVehicleCrossedLine(const ANSCENTER::Object& vehicle) {
    if (crossingLine_.size() != 2) return false;
    cv::Point center(vehicle.box.x + vehicle.box.width/2, vehicle.box.y + vehicle.box.height/2);
    double dist = cv::pointPolygonTest(crossingLine_, center, true);
    if (dist < 10.0 && std::find(crossedVehicleTrackIds_.begin(), crossedVehicleTrackIds_.end(), vehicle.trackId) == crossedVehicleTrackIds_.end()) {
        crossedVehicleTrackIds_.push_back(vehicle.trackId);
        return true;
    }
    return false;
}

int Vehicle::GetCrossedVehicleCount() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(crossedVehicleTrackIds_.size());
}

std::string Vehicle::ClassifyVehicleType(int classId) const {
    switch (classId) {
        case 0: return "car";
        case 1: return "motorbike";
        case 2: return "bus";
        case 3: return "truck";
        default: return "unknown";
    }
}

bool Vehicle::Destroy() {
    std::lock_guard<std::mutex> lock(mtx_);
    crossedVehicleTrackIds_.clear();
    return detector_.Destroy();
}