#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"

class TrafficLight {
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
    
    // Traffic light ROI
    std::vector<ANSCENTER::Region> m_trafficROIs;
    
    // Parameters
    std::vector<ANSCENTER::Params> m_parameters;

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