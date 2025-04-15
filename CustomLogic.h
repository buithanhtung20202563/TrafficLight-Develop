#ifndef CUSTOM_LOGIC_H
#define CUSTOM_LOGIC_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Vehicle.h"
#include "TrafficLight.h"
#include "ANSCustomTrafficLight.h"

class CustomLogic : public IANSCustomClass {
private:
    Vehicle m_vehicleDetector;
    TrafficLight m_trafficLightDetector;
    std::recursive_mutex m_mutex;

    // Store detected objects for visualization or further processing
    std::vector<CustomObject> m_lastDetectionResults;

    // Helper method to convert ANSCENTER::Object to CustomObject
    CustomObject ConvertToCustomObject(const ANSCENTER::Object& obj, const std::string& cameraId, int classIdOffset = 0);

public:
    CustomLogic();
    ~CustomLogic();

    // IANSCustomClass interface implementation
    bool Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) override;
    bool OptimizeModel(bool fp16) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& cameraId) override;
    bool ConfigureParamaters(std::vector<CustomParams>& param) override;
    bool Destroy() override;

    // Additional method for violation detection (used internally)
    void ProcessViolations(const std::vector<ANSCENTER::Object>& vehicles,
        const std::vector<ANSCENTER::Object>& trafficLights,
        const std::string& cameraId);
};

#endif // CUSTOM_LOGIC_H