#ifndef CUSTOM_LOGIC_H
#define CUSTOM_LOGIC_H

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include "Vehicle.h"
#include "TrafficLight.h"
#include "ANSLIB.h"

// Define the callback function type for triggers
typedef std::function<void(const std::string&, const cv::Mat&, const cv::Rect&)> ViolationCallback;

class CustomLogic {
private:
    // Components for detection
    Vehicle vehicleDetector;
    TrafficLight trafficLightDetector;
    
    // Configuration
    std::string modelDirectory;
    float detectionThreshold;
    
    // Callback for violations
    ViolationCallback violationCallback;
    
    // Internal state
    std::vector<int> violatingVehicles;
    bool isRedLight;
    
    // Mutex for thread safety
    std::recursive_mutex _mutex;
    
    // Helper methods
    bool isVehicleViolating(const ANSCENTER::Object& vehicle);
    void recordViolation(const ANSCENTER::Object& vehicle, const cv::Mat& frame);

public:
    CustomLogic();
    ~CustomLogic();
    
    // Initialization and configuration
    bool Initialize(const std::string& modelDir, float threshold);
    bool OptimizeModel(bool fp16);
    
    // Set callback for violation triggers
    void SetViolationCallback(ViolationCallback callback);
    
    // Process a frame
    bool ProcessFrame(const cv::Mat& frame, const std::string& cameraId);
    
    // Cleanup resources
    bool Destroy();
};

#endif // CUSTOM_LOGIC_H