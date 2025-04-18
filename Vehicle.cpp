#include "Vehicle.h"
#include <mutex>
#include <chrono>
#include "ANSCustomTrafficLight.h"

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

CACVehicle::CACVehicle() {
    m_sModelName = "vehicle";
    m_sClassName = "vehicle.names";
    m_nModelType = 4; // TensorRT model by default
    m_nDetectionType = 1; // Object detection
    m_fDetectionScoreThreshold = 0.4;
    m_fConfidenceThreshold = 0.5;
    m_fNMSThreshold = 0.5;
}

CACVehicle::~CACVehicle() {
    Destroy();
}

bool CACVehicle::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    m_sModelDirectory = modelDir;
    m_fDetectionScoreThreshold = threshold;

    // Check engine type and adjust model type if needed
    int engineType = m_cDetector.GetEngineType();
    if (engineType == 0) {
        // NVIDIA CPU - use ONNX model
        m_nModelType = 3;
    }

    // Load the CACVehicle detection model
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
    ConfigureParameters();

    return (result == 1);
}

bool CACVehicle::Optimize(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (m_cDetector.Optimize(fp16) == 1);
}

//bool CACVehicle::ConfigureParameters() {
//    // Create the detection area ROI based on the diagram
//    ANSCENTER::Region detectArea;
//    detectArea.regionType = 1; // Rectangle
//    detectArea.regionName = "DetectArea";
//
//    // Set the rectangle coordinates for the detection area
//    // These coordinates would match the "Detect Area" area shown in the diagram
//    detectArea.polygon.push_back(cv::Point(100, 200));
//    detectArea.polygon.push_back(cv::Point(400, 200));
//    detectArea.polygon.push_back(cv::Point(400, 400));
//    detectArea.polygon.push_back(cv::Point(100, 400));
//
//    m_vDetectAreaROI.clear();
//    m_vDetectAreaROI.push_back(detectArea);
//
//    // Create the crossing line ROI
//    ANSCENTER::Region crossingLine;
//    crossingLine.regionType = 2; // Line
//    crossingLine.regionName = "CrossingLine";
//
//    // Set the line coordinates for the crossing line
//    // These coordinates would match the "Crossing Line" shown in the diagram
//    crossingLine.polygon.push_back(cv::Point(100, 380));
//    crossingLine.polygon.push_back(cv::Point(400, 380));
//
//    m_vCrossingLineROI.clear();
//    m_vCrossingLineROI.push_back(crossingLine);
//
//    // Create the direction line ROI
//    ANSCENTER::Region directionLine;
//    directionLine.regionType = 4; // Direction line
//    directionLine.regionName = "Direction";
//
//    // Set the line coordinates for the direction line
//    directionLine.polygon.push_back(cv::Point(250, 350));
//    directionLine.polygon.push_back(cv::Point(250, 250));
//
//    directionLineROI.clear();
//    directionLineROI.push_back(directionLine);
//
//    // Create parameter structure
//    ANSCENTER::Params param;
//    param.handleId = 0; // Vehicle detector ID
//    param.handleName = modelName;
//
//    // Add threshold parameter
//    ANSCENTER::ParamType thresholdParam;
//    thresholdParam.type = 1; // double
//    thresholdParam.name = "threshold";
//    thresholdParam.value = std::to_string(detectionScoreThreshold);
//
//    param.handleParametersJson.push_back(thresholdParam);
//
//    // Add all ROIs to the parameter
//    param.ROIs.insert(param.ROIs.end(), m_vDetectAreaROI.begin(), m_vDetectAreaROI.end());
//    param.ROIs.insert(param.ROIs.end(), m_vCrossingLineROI.begin(), m_vCrossingLineROI.end());
//    param.ROIs.insert(param.ROIs.end(), directionLineROI.begin(), directionLineROI.end());
//
//    parameters.clear();
//    parameters.push_back(param);
//
//    return true;
//}

bool CACVehicle::ConfigureParameters()
{
    return true;
}

bool CACVehicle::SetParameters(const CustomParams& params) 
{
    m_stParameters = params;
     // Update ROIs if available

    if (params.handleId == 0) {
        m_vDetectAreaROI.clear();
        m_vCrossingLineROI.clear();
        m_vDirectionLineROI.clear();

        for (const auto& roi : params.ROIs) {
            if (roi.regionName == "DetectArea") {
                m_vDetectAreaROI.push_back(roi);
            }
            else if (roi.regionName == "CrossingLine") {
                m_vCrossingLineROI.push_back(roi);
            }
            else if (roi.regionName == "Direction") {
                m_vDirectionLineROI.push_back(roi);
            }
        }
    }
    return true;
}

CustomParams CACVehicle::GetParameters()
{
    return m_stParameters;
}

std::vector<ANSCENTER::Object> CACVehicle::DetectVehicles(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    std::vector<ANSCENTER::Object> detectedVehicles;
    try {
        // Run inference on the input image
        m_cDetector.RunInference(input, cameraId.c_str(), detectedVehicles);

        // Filter results to include only vehicles within the detection ROI
        if (!m_vDetectAreaROI.empty()) {
            std::vector<ANSCENTER::Object> filteredResults;

            for (const auto& roi : m_vDetectAreaROI) {
                // Check if the vehicle is within the detection area
                for (const auto& obj : detectedVehicles) {
                    // For rectangular ROIs
                    if (roi.regionType == 0 || roi.regionType == 1) {
                        // Convert polygon to rectangle for simple containment check
                        cv::Rect roiRect = cv::boundingRect(roi.polygon);
                        // Check if the vehicle's bounding box intersects with the ROI
						std::cout << "OBJECT: " << obj.box << std::endl;
                        if ((obj.box & roiRect).area() > 0) {
                            filteredResults.push_back(obj);
                        }
                    }
                }
            }

            // Update vehicle tracking with the filtered results
            UpdateVehicleTracking(filteredResults);

            return filteredResults;
        }

        // If no ROI filtering is applied, still update tracking
        UpdateVehicleTracking(detectedVehicles);

        return detectedVehicles;
    }
    catch (std::exception& e) {
        return std::vector<ANSCENTER::Object>();
    }
}

bool CACVehicle::IsVehicleCrossedLine(const ANSCENTER::Object& vehicle) {
    try {
        if (m_vDetectAreaROI.empty()) {
            return false;
        }

        // Get the detection area
        const auto& detectArea = m_vDetectAreaROI[0];
        std::vector<cv::Point> polygon = detectArea.polygon;
        
        // Get vehicle center point
        cv::Point vehicleCenter(
            vehicle.box.x + vehicle.box.width/2,
            vehicle.box.y + vehicle.box.height/2
        );

        // Check if point is inside polygon using OpenCV
        double result = cv::pointPolygonTest(polygon, vehicleCenter, true);
        
        // If point is inside or on the polygon (result >= 0)
        if (result >= 0) {
            // Check if vehicle is already tracked
            for (auto& tracked : trackedVehicles) {
                if (tracked.trackId == vehicle.trackId) {
                    if (!tracked.crossedLine) {
                        tracked.crossedLine = true;
                        return true;
                    }
                    return false;
                }
            }
            
            // If not tracked, add to tracking
            TrackedVehicle newVehicle;
            newVehicle.trackId = vehicle.trackId;
            newVehicle.lastPosition = vehicle.box;
            newVehicle.crossedLine = true;
            newVehicle.vehicleType = vehicle.className;
            newVehicle.lastSeen = std::chrono::system_clock::now();
            trackedVehicles.push_back(newVehicle);
            return true;
        }
        
        return false;
    }
    catch (const std::exception& e) {
        return false;
    }
}

void CACVehicle::UpdateVehicleTracking(const std::vector<ANSCENTER::Object>& vehicles) {
    auto currentTime = std::chrono::system_clock::now();

    // Update existing tracked vehicles
    for (auto& vehicle : vehicles) {
        bool found = false;

        for (auto& trackedVehicle : trackedVehicles) {
            if (trackedVehicle.trackId == vehicle.trackId) {
                // Update the vehicle
                trackedVehicle.lastPosition = vehicle.box;
                trackedVehicle.lastSeen = currentTime;
                trackedVehicle.vehicleType = vehicle.className;

                // Check if vehicle has crossed the line
                if (!trackedVehicle.crossedLine && IsVehicleCrossedLine(vehicle)) {
                    trackedVehicle.crossedLine = true;
                }

                found = true;
                break;
            }
        }

        // Add new vehicle to tracking list
        if (!found) {
            TrackedVehicle newVehicle;
            newVehicle.trackId = vehicle.trackId;
            newVehicle.lastPosition = vehicle.box;
            newVehicle.crossedLine = IsVehicleCrossedLine(vehicle);
            newVehicle.vehicleType = vehicle.className;
            newVehicle.lastSeen = currentTime;
            trackedVehicles.push_back(newVehicle);
        }
    }

    // Remove vehicles that haven't been seen for a while
    const auto MAX_AGE = std::chrono::seconds(5);
    trackedVehicles.erase(
        std::remove_if(
            trackedVehicles.begin(),
            trackedVehicles.end(),
            [currentTime, MAX_AGE](const TrackedVehicle& tv) {
                return currentTime - tv.lastSeen > MAX_AGE;
            }
        ),
        trackedVehicles.end()
    );
}

int CACVehicle::CountVehiclesCrossedLine() {
    int count = 0;
    for (const auto& vehicle : trackedVehicles) {
        if (vehicle.crossedLine) {
            count++;
        }
    }
    return count;
}

bool CACVehicle::IsCar(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "car" || vehicle.classId == 0;
}

bool CACVehicle::IsTruck(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "truck" || vehicle.classId == 3;
}

bool CACVehicle::IsBus(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "bus" || vehicle.classId == 2;
}

bool CACVehicle::IsMotorbike(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "motorbike" || vehicle.classId == 1;
}

bool CACVehicle::Destroy() {
    // Release resources
    trackedVehicles.clear();
    return true;
}
