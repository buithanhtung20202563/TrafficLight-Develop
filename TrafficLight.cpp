#include "TrafficLight.h"
#include <mutex>

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

CACTrafficLight::CACTrafficLight() {
    m_sModelName = "light";
    m_sClassName = "light.names";
    m_nModelType = 4; // TensorRT model by default
    m_nDetectionType = 1; // Object detection
    m_fDetectionScoreThreshold = 0.5;
    m_fConfidenceThreshold = 0.5;
    m_fNMSThreshold = 0.5;
}

CACTrafficLight::~CACTrafficLight() {
    Destroy();
}

bool CACTrafficLight::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    m_sModelDirectory = modelDir;
    m_fDetectionScoreThreshold = threshold;

    // Check engine type and adjust model type if needed
    int engineType = m_cDetector.GetEngineType();
    if (engineType == 0) {
        // NVIDIA CPU - use ONNX model
        m_nModelType = 3;
    }

    // Load the traffic light detection model
    std::string licenseKey = "";
    int result = m_cDetector.LoadModelFromFolder(
        licenseKey.c_str(),
        m_sModelName.c_str(),
        m_sClassName.c_str(),
        m_fDetectionScoreThreshold,
        m_fConfidenceThreshold,
        m_fNMSThreshold,
        1, // Auto detect engine
        m_nModelType,
        m_nDetectionType,
        m_sModelDirectory.c_str(),
        m_sLabelMap
    );

    // Configure default parameters
    // ConfigureParameters();

    return (result == 1);
}

bool CACTrafficLight::Optimize(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (m_cDetector.Optimize(fp16) == 1);
}
/*
***********************************************************************************
bool CACTrafficLight::ConfigureParameters() {
   // Create the traffic light ROI based on the diagram
   ANSCENTER::Region trafficRoi;
   trafficRoi.regionType = 1; // Rectangle
   trafficRoi.regionName = "TrafficRoi";

   // Set the rectangle coordinates for the traffic light detection area
   // These coordinates would match the "Traffic ROI" area shown in the diagram
   trafficRoi.polygon.push_back(cv::Point(100, 50));
   trafficRoi.polygon.push_back(cv::Point(300, 50));
   trafficRoi.polygon.push_back(cv::Point(300, 200));
   trafficRoi.polygon.push_back(cv::Point(100, 200));

   trafficROIs.clear();
   trafficROIs.push_back(trafficRoi);

   // Create parameter structure
   ANSCENTER::Params param;
   param.handleId = 1; // Traffic Light detector ID
   param.handleName = modelName;

   // Add threshold parameter
   ANSCENTER::ParamType thresholdParam;
   thresholdParam.type = 1; // double
   thresholdParam.name = "threshold";
   thresholdParam.value = std::to_string(detectionScoreThreshold);

   param.handleParametersJson.push_back(thresholdParam);
   param.ROIs = trafficROIs;

   parameters.clear();
   parameters.push_back(param);

   return true;
}
************************************************************************************ 
*/ 

bool CACTrafficLight::SetParameters(const CustomParams& params)
{
    m_stParameters = params;

    if (params.handleId == 1) {
        // Update ROIs if available
        for (const auto& roi : params.ROIs) {
            if (roi.regionName == "TrafficRoi") {
                m_vTrafficROIs.push_back(roi);
            }
        }
    }
    return true;
}

CustomParams CACTrafficLight::GetParameters()
{
	return m_stParameters;
}

std::vector<ANSCENTER::Object> CACTrafficLight::DetectTrafficLights(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    std::vector<ANSCENTER::Object> detectedLights;
    try {
        // Run inference on the input image
        m_cDetector.RunInference(input, cameraId.c_str(), detectedLights);

        // Filter results to include only objects within the traffic ROI
        if (!m_vTrafficROIs.empty()) {
            std::vector<ANSCENTER::Object> filteredResults;

            for (const auto& obj : detectedLights) {
                // Check if the object is within any of the ROIs
                for (const auto& roi : m_vTrafficROIs) {
                    // For rectangular ROIs
                    if (roi.regionType == 0 || roi.regionType == 1) {
                        // Convert polygon to rectangle for simple containment check
                        cv::Rect roiRect = cv::boundingRect(roi.polygon);
                        filteredResults.push_back(obj);
                    }
                }
            }

            return filteredResults;
        }

        return detectedLights;
    }
    catch (std::exception& e) {
        return std::vector<ANSCENTER::Object>();
    }
}

bool CACTrafficLight::IsGreen(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to green light (8 in the combined class IDs)
        // The combined labelMap has "green" at index 8
        if (light.className == "green" || light.classId == 8) {
            return true;
        }
    }
    return false;
}

bool CACTrafficLight::IsRed(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to red light (9 in the combined class IDs)
        // The combined labelMap has "red" at index 9
        if (light.className == "red" || light.classId == 9) {
            return true;
        }
    }
    return false;
}

bool CACTrafficLight::IsYellow(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to yellow light (10 in the combined class IDs)
        // The combined labelMap has "yellow" at index 10
        if (light.className == "yellow" || light.classId == 10) {
            return true;
        }
    }
    return false;
}

bool CACTrafficLight::Destroy() {
    // Release resources
    return true;
}