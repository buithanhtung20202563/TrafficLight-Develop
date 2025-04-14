#include "CustomLogic.h"
#include <algorithm>
#include <mutex>

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

CustomLogic::CustomLogic() 
    : m_vehicleDetectionThreshold(0.5),
      m_trafficLightDetectionThreshold(0.5),
      m_isRedLightOn(false)
{
    // Inherit from ANSCustomTL constructor
}

CustomLogic::~CustomLogic() 
{
    Destroy();
}

bool CustomLogic::Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) 
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    // Initialize base class
    bool baseInitialized = ANSCustomTL::Initialize(modelDirectory, detectionScoreThreshold, labelMap);
    
    // Initialize vehicle detector
    bool vehicleInitialized = m_vehicleDetector.Initialize(modelDirectory, m_vehicleDetectionThreshold);
    
    // Initialize traffic light detector
    bool trafficLightInitialized = m_trafficLightDetector.Initialize(modelDirectory, m_trafficLightDetectionThreshold);
    
    return baseInitialized && vehicleInitialized && trafficLightInitialized;
}

bool CustomLogic::OptimizeModel(bool fp16) 
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    // Optimize base models
    bool baseOptimized = ANSCustomTL::OptimizeModel(fp16);
    
    // Optimize vehicle detector
    bool vehicleOptimized = m_vehicleDetector.Optimize(fp16);
    
    // Optimize traffic light detector
    bool trafficLightOptimized = m_trafficLightDetector.Optimize(fp16);
    
    return baseOptimized && vehicleOptimized && trafficLightOptimized;
}

std::vector<CustomObject> CustomLogic::RunInference(const cv::Mat& input, const std::string& camera_id) 
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    // Run detection for vehicles
    std::vector<ANSCENTER::Object> detectedVehicles = m_vehicleDetector.DetectVehicles(input, camera_id);
    
    // Run detection for traffic lights
    std::vector<ANSCENTER::Object> detectedTrafficLights = m_trafficLightDetector.DetectTrafficLights(input, camera_id);
    
    // Convert ANSCENTER::Object to CustomObject for vehicles
    m_lastDetectedVehicles.clear();
    for (const auto& obj : detectedVehicles) {
        CustomObject customObj;
        customObj.classId = obj.classId;
        customObj.trackId = obj.trackId;
        customObj.className = obj.className;
        customObj.confidence = obj.confidence;
        customObj.box = obj.box;
        customObj.cameraId = camera_id;
        m_lastDetectedVehicles.push_back(customObj);
    }
    
    // Convert ANSCENTER::Object to CustomObject for traffic lights
    m_lastDetectedTrafficLights.clear();
    int maxVehicleClassId = 7; // Based on the class map in ANSCustomTrafficLight.cpp
    for (const auto& obj : detectedTrafficLights) {
        CustomObject customObj;
        customObj.classId = obj.classId + maxVehicleClassId;
        customObj.trackId = obj.trackId;
        customObj.className = obj.className;
        customObj.confidence = obj.confidence;
        customObj.box = obj.box;
        customObj.cameraId = camera_id;
        m_lastDetectedTrafficLights.push_back(customObj);
    }
    
    // Check for red light status
    m_isRedLightOn = IsRedLight(m_lastDetectedTrafficLights);
    
    // Process violations if there's a red light
    if (m_isRedLightOn) {
        ProcessViolations(m_lastDetectedVehicles, camera_id);
    }
    
    // Combine results (similar to ANSCustomTL::RunInference)
    std::vector<CustomObject> results;
    results.insert(results.end(), m_lastDetectedVehicles.begin(), m_lastDetectedVehicles.end());
    results.insert(results.end(), m_lastDetectedTrafficLights.begin(), m_lastDetectedTrafficLights.end());
    
    return results;
}

bool CustomLogic::ConfigureParamaters(std::vector<CustomParams>& param) 
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    // Use the base class configuration as a starting point
    bool baseConfigured = ANSCustomTL::ConfigureParamaters(param);
    
    // Configure vehicle detector parameters
    m_vehicleDetector.ConfigureParameters();
    
    // Configure traffic light detector parameters
    m_trafficLightDetector.ConfigureParameters();
    
    // Add additional custom parameters if needed
    if (!this->_param.empty()) {
        // Update the current parameters with any customizations
        for (auto& p : param) {
            if (p.handleId == 0) { // Vehicle detector
                // Add additional parameters specific to our violation detection logic
                CustomParamType violationParam;
                violationParam.type = 0; // int
                violationParam.name = "redLightViolationDetection";
                violationParam.value = "1"; // Enabled
                p.handleParametersJson.push_back(violationParam);
            }
        }
    }
    
    return baseConfigured;
}

bool CustomLogic::Destroy() 
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    // Clean up base class resources
    bool baseDestroyed = ANSCustomTL::Destroy();
    
    // Clean up vehicle detector
    bool vehicleDestroyed = m_vehicleDetector.Destroy();
    
    // Clean up traffic light detector
    bool trafficLightDestroyed = m_trafficLightDetector.Destroy();
    
    return baseDestroyed && vehicleDestroyed && trafficLightDestroyed;
}

bool CustomLogic::IsVehicleCrossingLine(const CustomObject& vehicle) 
{
    // Get crossing line information from vehicle detector
    if (this->_param.empty() || this->_param[0].ROIs.empty()) {
        return false;
    }
    
    // Find the crossing line ROI
    const CustomRegion* crossingLine = nullptr;
    for (const auto& roi : this->_param[0].ROIs) {
        if (roi.regionName == "CrossingLine") {
            crossingLine = &roi;
            break;
        }
    }
    
    if (!crossingLine || crossingLine->polygon.size() < 2) {
        return false;
    }
    
    // Line is defined by two points
    cv::Point lineStart = crossingLine->polygon[0];
    cv::Point lineEnd = crossingLine->polygon[1];
    
    // Get the center bottom point of the vehicle
    cv::Point vehiclePoint(
        vehicle.box.x + vehicle.box.width / 2,
        vehicle.box.y + vehicle.box.height
    );
    
    // Check if this point is close to the line
    // Calculate distance from point to line
    double lineLength = cv::norm(lineEnd - lineStart);
    double distance = std::abs((vehiclePoint.y - lineStart.y) * (lineEnd.x - lineStart.x) - 
                         (vehiclePoint.x - lineStart.x) * (lineEnd.y - lineStart.y)) / lineLength;
    
    // Consider the vehicle has crossed if it's within a certain threshold distance
    const double THRESHOLD_DISTANCE = 5.0;
    return distance < THRESHOLD_DISTANCE;
}

bool CustomLogic::IsRedLight(const std::vector<CustomObject>& trafficLights) 
{
    // Check if any traffic light is red
    for (const auto& light : trafficLights) {
        // The red light class ID is 9 in the combined class IDs
        // The combined labelMap has "red" at index 9 (based on ANSCustomTrafficLight.cpp)
        if (light.className == "red" || light.classId == 9) {
            return true;
        }
    }
    return false;
}

void CustomLogic::ProcessViolations(const std::vector<CustomObject>& vehicles, const std::string& cameraId) 
{
    // Check for vehicles crossing the line during red light
    for (const auto& vehicle : vehicles) {
        if (IsVehicleCrossingLine(vehicle)) {
            // We have a violation - vehicle crossing during red light
            if (m_violationCallback) {
                // Call the violation callback with the camera ID and violating vehicle
                m_violationCallback(cameraId, vehicle);
            }
        }
    }
}

void CustomLogic::SetViolationCallback(std::function<void(const std::string&, const CustomObject&)> callback) 
{
    this->m_violationCallback = callback;
}

void CustomLogic::SetDetectionThresholds(float vehicleThreshold, float trafficLightThreshold) 
{
    this->m_vehicleDetectionThreshold = vehicleThreshold;
    this->m_trafficLightDetectionThreshold = trafficLightThreshold;
}

std::vector<CustomObject> CustomLogic::GetLastDetectedVehicles() const 
{
    return m_lastDetectedVehicles;
}

std::vector<CustomObject> CustomLogic::GetLastDetectedTrafficLights() const 
{
    return m_lastDetectedTrafficLights;
}

bool CustomLogic::IsRedLightActive() const 
{
    return m_isRedLightOn;
}

std::vector<CustomObject> CustomLogic::DetectVehicles(const cv::Mat& input, const std::string& cameraId) 
{
    std::vector<ANSCENTER::Object> detectedVehicles = m_vehicleDetector.DetectVehicles(input, cameraId);
    
    // Convert ANSCENTER::Object to CustomObject
    std::vector<CustomObject> result;
    for (const auto& obj : detectedVehicles) {
        CustomObject customObj;
        customObj.classId = obj.classId;
        customObj.trackId = obj.trackId;
        customObj.className = obj.className;
        customObj.confidence = obj.confidence;
        customObj.box = obj.box;
        customObj.cameraId = cameraId;
        result.push_back(customObj);
    }
    
    return result;
}

std::vector<CustomObject> CustomLogic::DetectTrafficLights(const cv::Mat& input, const std::string& cameraId) 
{
    std::vector<ANSCENTER::Object> detectedLights = m_trafficLightDetector.DetectTrafficLights(input, cameraId);
    
    // Convert ANSCENTER::Object to CustomObject
    std::vector<CustomObject> result;
    int maxVehicleClassId = 7; // Based on the class map in ANSCustomTrafficLight.cpp
    for (const auto& obj : detectedLights) {
        CustomObject customObj;
        customObj.classId = obj.classId + maxVehicleClassId;
        customObj.trackId = obj.trackId;
        customObj.className = obj.className;
        customObj.confidence = obj.confidence;
        customObj.box = obj.box;
        customObj.cameraId = cameraId;
        result.push_back(customObj);
    }
    
    return result;
}