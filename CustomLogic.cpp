#include "CustomLogic.h"
#include <iostream>
#include <chrono>
#include <iomanip>

CustomLogic::CustomLogic() {
    // Constructor
}

CustomLogic::~CustomLogic() {
    Destroy();
}

CustomObject CustomLogic::ConvertToCustomObject(const ANSCENTER::Object& obj, const std::string& cameraId, int classIdOffset) {
    CustomObject customObj;
    customObj.classId = obj.classId + classIdOffset;
    customObj.trackId = obj.trackId;
    customObj.className = obj.className;
    customObj.confidence = obj.confidence;
    customObj.box = obj.box;
    customObj.cameraId = cameraId;
    customObj.polygon = obj.polygon;
    customObj.mask = obj.mask;
    customObj.kps = obj.kps;
    customObj.extraInfo = obj.extraInfo;
    return customObj;
}

bool CustomLogic::Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    _modelDirectory = modelDirectory;
    _detectionScoreThreshold = detectionScoreThreshold;

    // Initialize vehicle detector
    std::string vehicleModelDir = modelDirectory + "\\vehicle";
    bool vehicleInit = m_vehicleDetector.Initialize(vehicleModelDir, detectionScoreThreshold);
    if (!vehicleInit) {
        std::cerr << "Failed to initialize vehicle detector" << std::endl;
        return false;
    }

    // Initialize traffic light detector
    std::string trafficLightModelDir = modelDirectory + "\\light";
    bool trafficLightInit = m_trafficLightDetector.Initialize(trafficLightModelDir, detectionScoreThreshold);
    if (!trafficLightInit) {
        std::cerr << "Failed to initialize traffic light detector" << std::endl;
        return false;
    }

    // Create combined label map (similar to ANSCustomTL)
    labelMap = "car,motorbike,bus,truck,bike,container,tricycle,human,green,red,yellow";
    return true;
}

bool CustomLogic::OptimizeModel(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Optimize both detectors
    bool vehicleOpt = m_vehicleDetector.Optimize(fp16);
    bool trafficLightOpt = m_trafficLightDetector.Optimize(fp16);

    if (!vehicleOpt) {
        std::cerr << "Failed to optimize vehicle detector" << std::endl;
    }
    if (!trafficLightOpt) {
        std::cerr << "Failed to optimize traffic light detector" << std::endl;
    }

    return vehicleOpt && trafficLightOpt;
}

bool CustomLogic::ConfigureParamaters(std::vector<CustomParams>& param) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Configure vehicle detector parameters
    if (!m_vehicleDetector.ConfigureParameters()) {
        std::cerr << "Failed to configure vehicle detector parameters" << std::endl;
        return false;
    }

    // Configure traffic light detector parameters
    if (!m_trafficLightDetector.ConfigureParameters()) {
        std::cerr << "Failed to configure traffic light detector parameters" << std::endl;
        return false;
    }

    // Create parameter schema for CustomLogic
    if (_param.empty()) {
        // Vehicle detector parameters
        CustomParams vehicleParam;
        vehicleParam.handleId = 0;
        vehicleParam.handleName = "vehicle";

        CustomParamType modelTypeParam;
        modelTypeParam.type = 0; // int
        modelTypeParam.name = "modelType";
        modelTypeParam.value = "4"; // TensorRT
        vehicleParam.handleParametersJson.push_back(modelTypeParam);

        CustomParamType thresholdParam;
        thresholdParam.type = 1; // double
        thresholdParam.name = "threshold";
        thresholdParam.value = std::to_string(_detectionScoreThreshold);
        vehicleParam.handleParametersJson.push_back(thresholdParam);

        // Vehicle ROIs (Detect Area, Crossing Line, Direction)
        CustomRegion detectArea;
        detectArea.regionType = 1; // Rectangle
        detectArea.regionName = "DetectArea";
        detectArea.polygon = {
            cv::Point(100, 200), cv::Point(400, 200),
            cv::Point(400, 400), cv::Point(100, 400)
        };
        vehicleParam.ROIs.push_back(detectArea);

        CustomRegion crossingLine;
        crossingLine.regionType = 2; // Line
        crossingLine.regionName = "CrossingLine";
        crossingLine.polygon = { cv::Point(100, 380), cv::Point(400, 380) };
        vehicleParam.ROIs.push_back(crossingLine);

        CustomRegion directionLine;
        directionLine.regionType = 4; // Direction line
        directionLine.regionName = "Direction";
        directionLine.polygon = { cv::Point(250, 350), cv::Point(250, 250) };
        vehicleParam.ROIs.push_back(directionLine);

        _param.push_back(vehicleParam);

        // Traffic light detector parameters
        CustomParams lightParam;
        lightParam.handleId = 1;
        lightParam.handleName = "light";

        CustomParamType lightThresholdParam;
        lightThresholdParam.type = 1; // double
        lightThresholdParam.name = "threshold";
        lightThresholdParam.value = std::to_string(_detectionScoreThreshold);
        lightParam.handleParametersJson.push_back(lightThresholdParam);

        // Traffic light ROI
        CustomRegion trafficRoi;
        trafficRoi.regionType = 1; // Rectangle
        trafficRoi.regionName = "TrafficRoi";
        trafficRoi.polygon = {
            cv::Point(100, 50), cv::Point(300, 50),
            cv::Point(300, 200), cv::Point(100, 200)
        };
        lightParam.ROIs.push_back(trafficRoi);

        _param.push_back(lightParam);
    }

    // Return parameters to caller
    param = _param;
    return true;
}

std::vector<CustomObject> CustomLogic::RunInference(const cv::Mat& input) {
    return RunInference(input, "default_camera");
}

std::vector<CustomObject> CustomLogic::RunInference(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_lastDetectionResults.clear();

    try {
        // Run vehicle detection
        std::vector<ANSCENTER::Object> vehicles = m_vehicleDetector.DetectVehicles(input, cameraId);

        // Run traffic light detection
        std::vector<ANSCENTER::Object> trafficLights = m_trafficLightDetector.DetectTrafficLights(input, cameraId);

        // Process violations (logs to console)
        ProcessViolations(vehicles, trafficLights, cameraId);

        // Combine results
        for (const auto& obj : vehicles) {
            m_lastDetectionResults.push_back(ConvertToCustomObject(obj, cameraId));
        }

        // Adjust traffic light class IDs (offset by 8, assuming vehicle classes are 0-7)
        for (const auto& obj : trafficLights) {
            m_lastDetectionResults.push_back(ConvertToCustomObject(obj, cameraId, 8));
        }

        return m_lastDetectionResults;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in RunInference: " << e.what() << std::endl;
        return {};
    }
}

void CustomLogic::ProcessViolations(const std::vector<ANSCENTER::Object>& vehicles,
    const std::vector<ANSCENTER::Object>& trafficLights,
    const std::string& cameraId) {
    // Check if traffic light is red
    bool isRed = false;
    for (const auto& light : trafficLights) {
        if (light.className == "red" || light.classId == 1) { // Assuming red is classId 1 in TrafficLight
            isRed = true;
            break;
        }
    }

    if (isRed) {
        // Check for vehicles crossing the line
        for (const auto& vehicle : vehicles) {
            if (m_vehicleDetector.HasVehicleCrossedLine(vehicle)) {
                // Log violation information
                auto now = std::chrono::system_clock::now();
                auto now_c = std::chrono::system_clock::to_time_t(now);
                std::tm localTime;
                localtime_s(&localTime, &now_c); // Use localtime_s for thread-safe and secure conversion
                std::stringstream ss;
                ss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");


                std::cout << "Violation Detected!" << std::endl;
                std::cout << "Timestamp: " << ss.str() << std::endl;
                std::cout << "Camera ID: " << cameraId << std::endl;
                std::cout << "Vehicle Type: " << vehicle.className << std::endl;
                std::cout << "Track ID: " << vehicle.trackId << std::endl;
                std::cout << "Bounding Box: [x: " << vehicle.box.x
                    << ", y: " << vehicle.box.y
                    << ", width: " << vehicle.box.width
                    << ", height: " << vehicle.box.height << "]" << std::endl;
                std::cout << "Confidence: " << vehicle.confidence << std::endl;
                std::cout << "-----------------------------------" << std::endl;
            }
        }
    }
}

bool CustomLogic::Destroy() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Destroy both detectors
    bool vehicleDestroyed = m_vehicleDetector.Destroy();
    bool trafficLightDestroyed = m_trafficLightDetector.Destroy();

    if (!vehicleDestroyed) {
        std::cerr << "Failed to destroy vehicle detector" << std::endl;
    }
    if (!trafficLightDestroyed) {
        std::cerr << "Failed to destroy traffic light detector" << std::endl;
    }

    m_lastDetectionResults.clear();
    _param.clear();
    return vehicleDestroyed && trafficLightDestroyed;
}