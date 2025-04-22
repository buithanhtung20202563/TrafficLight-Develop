#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ANSLIB.h"
#include "ANSCustomData.h"

// TungBT: Modify member variable's name, local variable's name
// Class XYYZZ (Example class CACVehicle with X: Class, YY: Project, ZZ: Class name)
// Member variable: m_XY (Explain m_: member, X: varibale type, Y: variable's name)
class CACTrafficLight
{
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

    // Traffic light ROI
    std::vector<CustomRegion> m_vTrafficROIs;

    // Parameters
    CustomParams m_stParameters;

public:
    CACTrafficLight();
    ~CACTrafficLight();

    bool Initialize(const std::string &modelDir, float threshold);
    bool Optimize(bool fp16);
    // bool ConfigureParameters();
    bool SetParameters(const CustomParams &params);
    CustomParams GetParameters();

    std::vector<ANSCENTER::Object> DetectTrafficLights(const cv::Mat &input, const std::string &cameraId);

    bool IsGreen(const std::vector<ANSCENTER::Object> &detectedLights);
    bool IsRed(const std::vector<ANSCENTER::Object> &detectedLights);
    bool IsYellow(const std::vector<ANSCENTER::Object> &detectedLights);

    bool Destroy();
};

#endif // TRAFFIC_LIGHT_H