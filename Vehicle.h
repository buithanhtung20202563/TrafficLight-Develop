#ifndef VEHICLE_H
#define VEHICLE_H

#include <vector>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class Vehicle {
public:
    Vehicle();
    ~Vehicle();

    bool Initialize(const std::string& modelDir, float threshold);
    bool ConfigureParameters(const std::vector<cv::Point>& detectArea,
                            const std::vector<cv::Point>& crossingLine,
                            const std::vector<cv::Point>& directionLine);
    bool Optimize(bool fp16);

    // Detect and track vehicles in input frame
    std::vector<ANSCENTER::Object> DetectVehicles(const cv::Mat& input, const std::string& cameraId);

    // Check if a vehicle has crossed the crossing line
    bool HasVehicleCrossedLine(const ANSCENTER::Object& vehicle);

    // Count vehicles that have crossed the line
    int GetCrossedVehicleCount() const;

    // Classify vehicle type
    std::string ClassifyVehicleType(int classId) const;

    bool Destroy();

private:
    ANSCENTER::ANSLIB detector_;
    float threshold_;
    std::vector<cv::Point> detectArea_;
    std::vector<cv::Point> crossingLine_;
    std::vector<cv::Point> directionLine_;
    mutable std::mutex mtx_;

    // Tracking info
    std::vector<int> crossedVehicleTrackIds_;
};

#endif // VEHICLE_H