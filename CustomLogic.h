#ifndef CUSTOM_LOGIC_H
#define CUSTOM_LOGIC_H

#include "ANSCustomTrafficLight.h"
#include "Vehicle.h"
#include "TrafficLight.h"
#include <string>
#include <vector>
#include <functional>

class CustomLogic : public ANSCustomTL {
private:
    Vehicle m_vehicleDetector;
    TrafficLight m_trafficLightDetector;
    
    // Callback for violation detection
    std::function<void(const std::string&, const CustomObject&)> m_violationCallback;
    
    // Detection thresholds and parameters
    float m_vehicleDetectionThreshold;
    float m_trafficLightDetectionThreshold;
    
    // Internal tracking
    std::vector<CustomObject> m_lastDetectedVehicles;
    std::vector<CustomObject> m_lastDetectedTrafficLights;
    bool m_isRedLightOn;
    
    // Support functions
    bool IsVehicleCrossingLine(const CustomObject& vehicle);
    bool IsRedLight(const std::vector<CustomObject>& trafficLights);
    void ProcessViolations(const std::vector<CustomObject>& vehicles, const std::string& cameraId);
    
protected:
    // Overrides from ANSCustomTL
    bool Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) override;
    bool OptimizeModel(bool fp16) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& camera_id) override;
    bool ConfigureParamaters(std::vector<CustomParams>& param) override;
    bool Destroy() override;
    
public:
    CustomLogic();
    ~CustomLogic();
    
    // Violation detection function
    void SetViolationCallback(std::function<void(const std::string&, const CustomObject&)> callback);
    
    // Configuration methods
    void SetDetectionThresholds(float vehicleThreshold, float trafficLightThreshold);
    
    // Results access
    std::vector<CustomObject> GetLastDetectedVehicles() const;
    std::vector<CustomObject> GetLastDetectedTrafficLights() const;
    bool IsRedLightActive() const;
    
    // Direct detection methods (exposed for convenience)
    std::vector<CustomObject> DetectVehicles(const cv::Mat& input, const std::string& cameraId);
    std::vector<CustomObject> DetectTrafficLights(const cv::Mat& input, const std::string& cameraId);
};

#endif // CUSTOM_LOGIC_H