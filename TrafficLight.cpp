#include "TrafficLight.h"
#include <mutex>
#include <opencv2/opencv.hpp>

TrafficLight::TrafficLight() : threshold_(0.5f), mtx_() {}
TrafficLight::~TrafficLight() { Destroy(); }

bool TrafficLight::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::mutex> lock(mtx_);
    threshold_ = threshold;
    return detector_.Initialize(modelDir.c_str(), threshold_);
}

bool TrafficLight::ConfigureParameters(const std::vector<cv::Point>& trafficRoi) {
    std::lock_guard<std::mutex> lock(mtx_);
    trafficRoi_ = trafficRoi;
    return true;
}

bool TrafficLight::Optimize(bool fp16) {
    std::lock_guard<std::mutex> lock(mtx_);
    return detector_.Optimize(fp16);
}

std::vector<ANSCENTER::Object> TrafficLight::DetectTrafficLights(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ANSCENTER::Object> detected;
    detector_.RunInference(input, cameraId.c_str(), detected);
    // TODO: Filter by trafficRoi_ if needed
    return detected;
}

bool TrafficLight::IsRed(const std::vector<ANSCENTER::Object>& detectedLights) const {
    for (const auto& obj : detectedLights) {
        if (obj.className == "red" || obj.classId == 1) return true;
    }
    return false;
}
bool TrafficLight::IsGreen(const std::vector<ANSCENTER::Object>& detectedLights) const {
    for (const auto& obj : detectedLights) {
        if (obj.className == "green" || obj.classId == 0) return true;
    }
    return false;
}
bool TrafficLight::IsYellow(const std::vector<ANSCENTER::Object>& detectedLights) const {
    for (const auto& obj : detectedLights) {
        if (obj.className == "yellow" || obj.classId == 2) return true;
    }
    return false;
}

bool TrafficLight::Destroy() {
    std::lock_guard<std::mutex> lock(mtx_);
    return detector_.Destroy();
}