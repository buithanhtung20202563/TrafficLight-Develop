#ifndef VEHICLE_H
#define VEHICLE_H
#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"
#include "ANSCustomData.h"

// TungBT: Modify member variable's name, local variable's name
// Class XYYZZ (Example class CACVehicle with X: Class, YY: Project, ZZ: Class name)
// Member variable: m_XY (Explain m_: member, X: varibale type, Y: variable's name)
class CACVehicle {
private:
    ANSCENTER::ANSLIB m_cDetector;
    std::string m_sModelName;
    std::string m_sClassName;
    int m_nModelType;
    int m_nDetectionType;
    std::string m_sModelDirectory;
    float m_fDetectionScoreThreshold;
    float m_fConfidenceThreshold;
    float m_fNMSThreshold;
    std::string m_sLabelMap;

    // Vehicle-related ROIs
    std::vector<CustomRegion> m_vDetectAreaROI;
    std::vector<CustomRegion> m_vCrossingLineROI;
    std::vector<CustomRegion> m_vDirectionLineROI;

    // Parameters
    CustomParams m_stParameters;

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
    CACVehicle();
    ~CACVehicle();

    bool Initialize(const std::string& modelDir, float threshold);
    bool Optimize(bool fp16);
    bool ConfigureParameters();
    bool SetParameters(const CustomParams& params);
    CustomParams GetParameters();
	std::vector<CustomRegion> GetDetectAreaROI() const { return m_vDetectAreaROI; }
	std::vector<CustomRegion> GetCrossingLineROI() const { return m_vCrossingLineROI; }
	std::vector<CustomRegion> GetDirectionLineROI() const { return m_vDirectionLineROI; }

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