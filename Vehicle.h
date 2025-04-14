#ifndef VEHICLE_H
#define VEHICLE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class Vehicle {
private:
    ANSCENTER::ANSLIB m_detector;
    std::string m_modelName;
    std::string m_className;
    int m_modelType;
    int m_detectionType;
    std::string m_modelDirectory;
    float m_detectionScoreThreshold;
    float m_confidenceThreshold;
    float m_nmsThreshold;
    std::string m_labelMap;
    
    // Vehicle-related ROIs
    std::vector<ANSCENTER::Region> m_detectAreaROI;
    std::vector<ANSCENTER::Region> m_crossingLineROI;
    std::vector<ANSCENTER::Region> m_directionLineROI;
    
    // Parameters
    std::vector<ANSCENTER::Params> m_parameters;
    
    // Tracking data
    struct TrackedVehicle {
        int trackId;
        cv::Rect lastPosition;
        bool crossedLine;
        std::string vehicleType;
        std::chrono::system_clock::time_point lastSeen;
    };
    
    std::vector<TrackedVehicle> m_trackedVehicles;
    
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