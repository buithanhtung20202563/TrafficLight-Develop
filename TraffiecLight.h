#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class TrafficLight {
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
    
    // Traffic light ROI
    std::vector<ANSCENTER::Region> trafficROIs;
    
    // Parameters
    std::vector<ANSCENTER::Params> parameters;

public:
    TrafficLight();
    ~TrafficLight();
    
    bool Initialize(const std::string& modelDir, float threshold);
    bool Optimize(bool fp16);
    bool ConfigureParameters();
    bool SetParameters(const std::vector<ANSCENTER::Params>& params);
    
    std::vector<ANSCENTER::Object> DetectTrafficLights(const cv::Mat& input, const std::string& cameraId);
    
    bool IsGreen(const std::vector<ANSCENTER::Object>& detectedLights);
    bool IsRed(const std::vector<ANSCENTER::Object>& detectedLights);
    bool IsYellow(const std::vector<ANSCENTER::Object>& detectedLights);
    
    bool Destroy();
};

#endif // TRAFFIC_LIGHT_H