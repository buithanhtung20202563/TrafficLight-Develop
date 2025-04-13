#ifndef VEHICLE_H
#define VEHICLE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class Vehicle {
private:
    ANSCENTER::ANSLIB detector;
    std::string modelName;
    std::string className;
    int modelType;
    int detectionType;
    std::string modelDirectory;
    float detectionScoreThreshold;
    float confidenceThreshold;
    float nmsThreshold;
    std::string labelMap;
    
    // Vehicle-related ROIs
    std::vector<ANSCENTER::Region> detectAreaROI;
    std::vector<ANSCENTER::Region> crossingLineROI;
    std::vector<ANSCENTER::Region> directionLineROI;
    
    // Parameters
    std::vector<ANSCENTER::Params> parameters;
    
    // Tracking data
    struct TrackedVehicle {
        int trackId;
        cv::Rect lastPosition;
        bool crossedLine;
        std::string vehicleType;
        std::chrono::system_clock::time_point lastSeen;
    };
    
    std::vector<TrackedVehicle> trackedVehicles;
    
public:
    Vehicle();
    ~Vehicle();
    
    bool Initialize(const std::string& modelDir, float threshold);
    bool Optimize(bool fp16);
    bool ConfigureParameters();
    bool SetParameters(const std::vector<ANSCENTER::Params>& params);
    
    std::vector<ANSCENTER::Object> DetectVehicles(const cv::Mat& input, const std::string& cameraId);
    
    // Methods for line crossing detection
    bool HasVehicleCrossedLine(const ANSCENTER::Object& vehicle);
    int CountVehiclesCrossedLine();
    void UpdateVehicleTracking(const std::vector<ANSCENTER::Object>& vehicles);
    
    // Vehicle classification methods
    bool IsCar(const ANSCENTER::Object& vehicle);
    bool IsTruck(const ANSCENTER::Object& vehicle);
    bool IsBus(const ANSCENTER::Object& vehicle);
    bool IsMotorbike(const ANSCENTER::Object& vehicle);
    
    bool Destroy();
};

#endif // VEHICLE_H