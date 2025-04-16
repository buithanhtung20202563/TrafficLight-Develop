#ifndef CUSTOM_LOGIC_H
#define CUSTOM_LOGIC_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "Vehicle.h"
#include "TrafficLight.h"
#include "ANSCustomTrafficLight.h"

struct ViolationInfo {
    ANSCENTER::Object vehicle;
    std::string cameraId;
    std::string timestamp;
};

class CustomLogic {
public:
    CustomLogic();
    ~CustomLogic();

    bool Initialize(const std::string& modelDir, float threshold);
    bool ConfigureParameters();
    bool OptimizeModel(bool fp16);

    // Main inference: returns detected vehicles and traffic lights
    void RunInference(const cv::Mat& input, const std::string& cameraId,
                      std::vector<ANSCENTER::Object>& vehicles,
                      std::vector<ANSCENTER::Object>& trafficLights);

    // Detect and log violations (vehicle crosses line when red light)
    void ProcessViolations(const std::vector<ANSCENTER::Object>& vehicles,
                           const std::vector<ANSCENTER::Object>& trafficLights,
                           const std::string& cameraId);

    bool Destroy();

private:
    Vehicle vehicleModule_;
    TrafficLight trafficLightModule_;
    std::shared_ptr<ANSCustomTL> customTLModule_;
};

#endif // CUSTOM_LOGIC_H