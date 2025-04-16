#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <vector>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class TrafficLight {
public:
    TrafficLight();
    ~TrafficLight();

    bool Initialize(const std::string& modelDir, float threshold);
    bool ConfigureParameters(const std::vector<cv::Point>& trafficRoi);
    bool Optimize(bool fp16);

    // Detect traffic lights in input frame
    std::vector<ANSCENTER::Object> DetectTrafficLights(const cv::Mat& input, const std::string& cameraId);

    // Get traffic light state
    bool IsRed(const std::vector<ANSCENTER::Object>& detectedLights) const;
    bool IsGreen(const std::vector<ANSCENTER::Object>& detectedLights) const;
    bool IsYellow(const std::vector<ANSCENTER::Object>& detectedLights) const;

    bool Destroy();

private:
    ANSCENTER::ANSLIB detector_;
    float threshold_;
    std::vector<cv::Point> trafficRoi_;
    mutable std::mutex mtx_;
};

#endif // TRAFFIC_LIGHT_H