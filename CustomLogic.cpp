#include "CustomLogic.h"
#include <mutex>
#include <chrono>
#include <iostream>

CustomLogic::CustomLogic() : isRedLight(false) {
    // Initialize internal state
}

CustomLogic::~CustomLogic() {
    Destroy();
}

bool CustomLogic::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    
    modelDirectory = modelDir;
    detectionThreshold = threshold;
    
    // Initialize vehicle detector
    bool vehicleResult = vehicleDetector.Initialize(modelDirectory, detectionThreshold);
    
    // Initialize traffic light detector
    bool trafficResult = trafficLightDetector.Initialize(modelDirectory, detectionThreshold);
    
    // Return true only if both initializations succeeded
    return vehicleResult && trafficResult;
}

bool CustomLogic::OptimizeModel(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    
    // Optimize both detectors
    bool vehicleOptimized = vehicleDetector.Optimize(fp16);
    bool trafficOptimized = trafficLightDetector.Optimize(fp16);
    
    return vehicleOptimized && trafficOptimized;
}

void CustomLogic::SetViolationCallback(ViolationCallback callback) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    violationCallback = callback;
}

bool CustomLogic::ProcessFrame(const cv::Mat& frame, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    
    try {
        // Step 1: Detect traffic lights
        std::vector<ANSCENTER::Object> detectedLights = trafficLightDetector.DetectTrafficLights(frame, cameraId);
        
        // Update red light state
        isRedLight = trafficLightDetector.IsRed(detectedLights);
        
        // If it's not a red light, no violations can occur
        if (!isRedLight) {
            return true;
        }
        
        // Step 2: Detect vehicles
        std::vector<ANSCENTER::Object> detectedVehicles = vehicleDetector.DetectVehicles(frame, cameraId);
        
        // Step 3: Process each vehicle to check for violations
        for (const auto& vehicle : detectedVehicles) {
            // Check if the vehicle has crossed the line
            if (vehicleDetector.HasVehicleCrossedLine(vehicle)) {
                // Only record if it's a new violation (not already in our list)
                if (std::find(violatingVehicles.begin(), violatingVehicles.end(), vehicle.trackId) == violatingVehicles.end()) {
                    // Record the violation
                    recordViolation(vehicle, frame);
                    
                    // Add to our list so we don't trigger again for the same vehicle
                    violatingVehicles.push_back(vehicle.trackId);
                }
            }
        }
        
        // Clean up old violation records (vehicles that are no longer tracked)
        // This prevents our list from growing indefinitely
        auto it = violatingVehicles.begin();
        while (it != violatingVehicles.end()) {
            bool stillPresent = false;
            for (const auto& vehicle : detectedVehicles) {
                if (vehicle.trackId == *it) {
                    stillPresent = true;
                    break;
                }
            }
            
            if (!stillPresent) {
                it = violatingVehicles.erase(it);
            } else {
                ++it;
            }
        }
        
        return true;
    }
    catch (std::exception& e) {
        std::cerr << "Error in ProcessFrame: " << e.what() << std::endl;
        return false;
    }
}

bool CustomLogic::isVehicleViolating(const ANSCENTER::Object& vehicle) {
    // A vehicle is violating if it crosses the line while the traffic light is red
    return isRedLight && vehicleDetector.HasVehicleCrossedLine(vehicle);
}

void CustomLogic::recordViolation(const ANSCENTER::Object& vehicle, const cv::Mat& frame) {
    // Create a violation message
    std::string violationType = "RED_LIGHT_VIOLATION";
    std::string vehicleType = vehicle.className;
    
    std::string message = "Violation detected: " + violationType + 
                         ", Vehicle type: " + vehicleType + 
                         ", Track ID: " + std::to_string(vehicle.trackId);
    
    // If a callback is registered, trigger it
    if (violationCallback) {
        violationCallback(message, frame, vehicle.box);
    }
    
    // Log the violation
    std::cout << "[" << std::chrono::system_clock::now() << "] " << message << std::endl;
}

bool CustomLogic::Destroy() {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    
    // Clean up resources
    bool vehicleDestroyed = vehicleDetector.Destroy();
    bool trafficDestroyed = trafficLightDetector.Destroy();
    
    // Clear internal state
    violatingVehicles.clear();
    
    return vehicleDestroyed && trafficDestroyed;
}