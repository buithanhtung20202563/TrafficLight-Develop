#include "CustomLogic.h"
#include <iostream>
#include <chrono>
#include <iomanip>

CustomLogic::CustomLogic() {
    m_ansCustomTL = std::make_shared<ANSCustomTL>();
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

    // Initialize ANSCustomTL
    bool customTLInit = m_ansCustomTL->Initialize(modelDirectory, detectionScoreThreshold, labelMap);
    if (!customTLInit) {
        std::cerr << "Failed to initialize ANSCustomTL" << std::endl;
        return false;
    }

    // Create combined label map (similar to ANSCustomTL)
    labelMap = "car,motorbike,bus,truck,bike,container,tricycle,human,green,red,yellow";
    return true;
}

bool CustomLogic::OptimizeModel(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Optimize all detectors
    bool vehicleOpt = m_vehicleDetector.Optimize(fp16);
    bool trafficLightOpt = m_trafficLightDetector.Optimize(fp16);
    bool customTLOpt = m_ansCustomTL->OptimizeModel(fp16);

    if (!vehicleOpt) {
        std::cerr << "Failed to optimize vehicle detector" << std::endl;
    }
    if (!trafficLightOpt) {
        std::cerr << "Failed to optimize traffic light detector" << std::endl;
    }
    if (!customTLOpt) {
        std::cerr << "Failed to optimize ANSCustomTL" << std::endl;
    }

    return vehicleOpt && trafficLightOpt && customTLOpt;
}

bool CustomLogic::ConfigureParamaters(std::vector<CustomParams>& param) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    bool ok1 = m_vehicleDetector.ConfigureParameters();
    bool ok2 = m_trafficLightDetector.ConfigureParameters();
    bool ok3 = m_ansCustomTL->ConfigureParamaters(param);
    return ok1 && ok2 && ok3;
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

std::vector<CustomObject> CustomLogic::RunCustomTLInference(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_ansCustomTL->RunInference(input, cameraId);
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
    // Destroy all detectors
    bool vehicleDestroyed = m_vehicleDetector.Destroy();
    bool trafficLightDestroyed = m_trafficLightDetector.Destroy();
    bool customTLDestroyed = m_ansCustomTL->Destroy();

    if (!vehicleDestroyed) {
        std::cerr << "Failed to destroy vehicle detector" << std::endl;
    }
    if (!trafficLightDestroyed) {
        std::cerr << "Failed to destroy traffic light detector" << std::endl;
    }
    if (!customTLDestroyed) {
        std::cerr << "Failed to destroy ANSCustomTL" << std::endl;
    }

    m_lastDetectionResults.clear();
    _param.clear();
    return vehicleDestroyed && trafficLightDestroyed && customTLDestroyed;
}

void CustomLogic::DetectRedLightViolationsFromDir(const std::string& directoryPath) {
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            // Chỉ xử lý file ảnh hoặc video
            if (filePath.ends_with(".jpg") || filePath.ends_with(".png") || filePath.ends_with(".bmp") || filePath.ends_with(".jpeg")) {
                cv::Mat img = cv::imread(filePath);
                if (!img.empty()) {
                    auto results = RunInference(img);
                    // Kiểm tra vi phạm: có phương tiện trong vùng crossing line khi đèn đỏ
                    bool redLight = false;
                    for (const auto& obj : results) {
                        if (obj.className == "red" || obj.classId == 9) {
                            redLight = true;
                            break;
                        }
                    }
                    if (redLight) {
                        for (const auto& obj : results) {
                            // Giả sử crossing line là một vùng ROI, ở đây chỉ log ra phương tiện khi có đèn đỏ
                            if (obj.classId >= 0 && obj.classId <= 7) { // classId phương tiện
                                std::cout << "[VIOLATION] Vehicle " << obj.className << " (trackId: " << obj.trackId << ") crossed line during RED light in file: " << filePath << std::endl;
                            }
                        }
                    }
                }
            }
            // Có thể mở rộng xử lý video nếu muốn
        }
    }
}