#include "TrafficLight.h"
#include <mutex>

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

TrafficLight::TrafficLight() {
    modelName = "light";
    className = "light.names";
    modelType = 4; // TensorRT model by default
    detectionType = 1; // Object detection
    detectionScoreThreshold = 0.5;
    confidenceThreshold = 0.5;
    nmsThreshold = 0.5;
}

TrafficLight::~TrafficLight() {
    Destroy();
}

bool TrafficLight::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    modelDirectory = modelDir;
    detectionScoreThreshold = threshold;

    // Check engine type and adjust model type if needed
    int engineType = detector.GetEngineType();
    if (engineType == 0) {
        // NVIDIA CPU - use ONNX model
        modelType = 3;
    }

    // Load the traffic light detection model
    std::string licenseKey = "";
    int result = detector.LoadModelFromFolder(
        licenseKey.c_str(),
        modelName.c_str(),
        className.c_str(),
        detectionScoreThreshold,
        confidenceThreshold,
        nmsThreshold,
        1, // Auto detect engine
        modelType,
        detectionType,
        modelDirectory.c_str(),
        labelMap
    );

    // Configure default parameters
    ConfigureParameters();

    return (result == 1);
}

bool TrafficLight::Optimize(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (detector.Optimize(fp16) == 1);
}

bool TrafficLight::ConfigureParameters() {
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

bool TrafficLight::SetParameters(const std::vector<ANSCENTER::Params>& params) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (params.empty()) {
        return false;
    }

    parameters.clear();
    for (const auto& p : params) {
        parameters.push_back(p);
    }

    // Update ROIs if available
    for (const auto& param : parameters) {
        if (param.handleId == 1 && !param.ROIs.empty()) {
            trafficROIs = param.ROIs;
            break;
        }
    }

    return true;
}

std::vector<ANSCENTER::Object> TrafficLight::DetectTrafficLights(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    std::vector<ANSCENTER::Object> detectedLights;
    try {
        // Run inference on the input image
        detector.RunInference(input, cameraId.c_str(), detectedLights);

        // Filter results to include only objects within the traffic ROI
        if (!trafficROIs.empty()) {
            std::vector<ANSCENTER::Object> filteredResults;

            for (const auto& obj : detectedLights) {
                // Check if the object is within any of the ROIs
                for (const auto& roi : trafficROIs) {
                    // For rectangular ROIs
                    if (roi.regionType == 0 || roi.regionType == 1) {
                        // Convert polygon to rectangle for simple containment check
                        cv::Rect roiRect = cv::boundingRect(roi.polygon);

                        // Check if the object's bounding box intersects with the ROI
                        if ((obj.box & roiRect).area() > 0) {
                            filteredResults.push_back(obj);
                            break;
                        }
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

bool TrafficLight::IsGreen(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to green light (8 in the combined class IDs)
        // The combined labelMap has "green" at index 8
        if (light.className == "green" || light.classId == 8) {
            return true;
        }
    }
    return false;
}

bool TrafficLight::IsRed(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to red light (9 in the combined class IDs)
        // The combined labelMap has "red" at index 9
        if (light.className == "red" || light.classId == 9) {
            return true;
        }
    }
    return false;
}

bool TrafficLight::IsYellow(const std::vector<ANSCENTER::Object>& detectedLights) {
    for (const auto& light : detectedLights) {
        // Check if classId corresponds to yellow light (10 in the combined class IDs)
        // The combined labelMap has "yellow" at index 10
        if (light.className == "yellow" || light.classId == 10) {
            return true;
        }
    }
    return false;
}

bool TrafficLight::Destroy() {
    // Release resources
    return true;
}