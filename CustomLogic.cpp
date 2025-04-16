#include "CustomLogic.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <memory>

CustomLogic::CustomLogic() : 
    vehicleModule_(),
    trafficLightModule_(),
    customTLModule_(std::make_shared<ANSCustomTL>()) 
{
}

CustomLogic::~CustomLogic() { Destroy(); }

bool CustomLogic::Initialize(const std::string& modelDir, float threshold) {
    bool ok1 = vehicleModule_.Initialize(modelDir + "\\vehicle", threshold);
    bool ok2 = trafficLightModule_.Initialize(modelDir + "\\light", threshold);
    std::string dummyLabelMap;
    bool ok3 = customTLModule_->Initialize(modelDir, threshold, dummyLabelMap);
    return ok1 && ok2 && ok3;
}

bool CustomLogic::ConfigureParameters() {
    // Example ROI (bạn có thể thay bằng vùng thực tế)
    std::vector<cv::Point> detectArea = { {100,200},{400,200},{400,400},{100,400} };
    std::vector<cv::Point> crossingLine = { {100,380},{400,380} };
    std::vector<cv::Point> directionLine = { {250,350},{250,250} };
    std::vector<cv::Point> trafficRoi = { {100,50},{300,50},{300,200},{100,200} };

    bool ok1 = vehicleModule_.ConfigureParameters(detectArea, crossingLine, directionLine);
    bool ok2 = trafficLightModule_.ConfigureParameters(trafficRoi);
    return ok1 && ok2;
}

bool CustomLogic::OptimizeModel(bool fp16) {
    bool ok1 = vehicleModule_.Optimize(fp16);
    bool ok2 = trafficLightModule_.Optimize(fp16);
    bool ok3 = customTLModule_->OptimizeModel(fp16);
    return ok1 && ok2 && ok3;
}

void CustomLogic::RunInference(const cv::Mat& input, const std::string& cameraId,
                               std::vector<ANSCENTER::Object>& vehicles,
                               std::vector<ANSCENTER::Object>& trafficLights) {
    vehicles = vehicleModule_.DetectVehicles(input, cameraId);
    trafficLights = trafficLightModule_.DetectTrafficLights(input, cameraId);
}

void CustomLogic::ProcessViolations(const std::vector<ANSCENTER::Object>& vehicles,
                                    const std::vector<ANSCENTER::Object>& trafficLights,
                                    const std::string& cameraId) {
    bool red = trafficLightModule_.IsRed(trafficLights);
    if (!red) return;
    for (const auto& v : vehicles) {
        if (vehicleModule_.HasVehicleCrossedLine(v)) {
            // Log violation
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::cout << "[VIOLATION] Vehicle " << v.className
                      << " (trackId: " << v.trackId << ") crossed line during RED light. "
                      << "Camera: " << cameraId
                      << " Time: " << std::put_time(std::localtime(&now_c), "%F %T")
                      << std::endl;
        }
    }
}

bool CustomLogic::Destroy() {
    bool ok1 = vehicleModule_.Destroy();
    bool ok2 = trafficLightModule_.Destroy();
    bool ok3 = customTLModule_->Destroy();
    return ok1 && ok2 && ok3;
}