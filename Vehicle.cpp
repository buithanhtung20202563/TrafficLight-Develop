#include "Vehicle.h"
#include <mutex>
#include <chrono>

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

Vehicle::Vehicle() {
    modelName = "vehicle";
    className = "vehicle.names";
    modelType = 4; // TensorRT model by default
    detectionType = 1; // Object detection
    detectionScoreThreshold =
        detectionScoreThreshold = 0.5;
    confidenceThreshold = 0.5;
    nmsThreshold = 0.5;
}

Vehicle::~Vehicle() {
    Destroy();
}

bool Vehicle::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    modelDirectory = modelDir;
    detectionScoreThreshold = threshold;

    // Check engine type and adjust model type if needed
    int engineType = detector.GetEngineType();
    if (engineType == 0) {
        // NVIDIA CPU - use ONNX model
        modelType = 3;
    }

    // Load the vehicle detection model
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

bool Vehicle::Optimize(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (detector.Optimize(fp16) == 1);
}

bool Vehicle::ConfigureParameters() {
    // Create the detection area ROI based on the diagram
    ANSCENTER::Region detectArea;
    detectArea.regionType = 1; // Rectangle
    detectArea.regionName = "DetectArea";

    // Set the rectangle coordinates for the detection area
    // These coordinates would match the "Detect Area" area shown in the diagram
    detectArea.polygon.push_back(cv::Point(100, 200));
    detectArea.polygon.push_back(cv::Point(400, 200));
    detectArea.polygon.push_back(cv::Point(400, 400));
    detectArea.polygon.push_back(cv::Point(100, 400));

    detectAreaROI.clear();
    detectAreaROI.push_back(detectArea);

    // Create the crossing line ROI
    ANSCENTER::Region crossingLine;
    crossingLine.regionType = 2; // Line
    crossingLine.regionName = "CrossingLine";

    // Set the line coordinates for the crossing line
    // These coordinates would match the "Crossing Line" shown in the diagram
    crossingLine.polygon.push_back(cv::Point(100, 380));
    crossingLine.polygon.push_back(cv::Point(400, 380));

    crossingLineROI.clear();
    crossingLineROI.push_back(crossingLine);

    // Create the direction line ROI
    ANSCENTER::Region directionLine;
    directionLine.regionType = 4; // Direction line
    directionLine.regionName = "Direction";

    // Set the line coordinates for the direction line
    directionLine.polygon.push_back(cv::Point(250, 350));
    directionLine.polygon.push_back(cv::Point(250, 250));

    directionLineROI.clear();
    directionLineROI.push_back(directionLine);

    // Create parameter structure
    ANSCENTER::Params param;
    param.handleId = 0; // Vehicle detector ID
    param.handleName = modelName;

    // Add threshold parameter
    ANSCENTER::ParamType thresholdParam;
    thresholdParam.type = 1; // double
    thresholdParam.name = "threshold";
    thresholdParam.value = std::to_string(detectionScoreThreshold);

    param.handleParametersJson.push_back(thresholdParam);

    // Add all ROIs to the parameter
    param.ROIs.insert(param.ROIs.end(), detectAreaROI.begin(), detectAreaROI.end());
    param.ROIs.insert(param.ROIs.end(), crossingLineROI.begin(), crossingLineROI.end());
    param.ROIs.insert(param.ROIs.end(), directionLineROI.begin(), directionLineROI.end());

    parameters.clear();
    parameters.push_back(param);

    return true;
}

bool Vehicle::SetParameters(const std::vector<ANSCENTER::Params>& params) {
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
        if (param.handleId == 0) {
            detectAreaROI.clear();
            crossingLineROI.clear();
            directionLineROI.clear();

            for (const auto& roi : param.ROIs) {
                if (roi.regionName == "DetectArea") {
                    detectAreaROI.push_back(roi);
                }
                else if (roi.regionName == "CrossingLine") {
                    crossingLineROI.push_back(roi);
                }
                else if (roi.regionName == "Direction") {
                    directionLineROI.push_back(roi);
                }
            }
            break;
        }
    }

    return true;
}

std::vector<ANSCENTER::Object> Vehicle::DetectVehicles(const cv::Mat& input, const std::string& cameraId) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    std::vector<ANSCENTER::Object> detectedVehicles;
    try {
        // Run inference on the input image
        detector.RunInference(input, cameraId.c_str(), detectedVehicles);

        // Filter results to include only vehicles within the detection ROI
        if (!detectAreaROI.empty()) {
            std::vector<ANSCENTER::Object> filteredResults;

            for (const auto& obj : detectedVehicles) {
                // Check if the vehicle is within the detection area
                for (const auto& roi : detectAreaROI) {
                    // For rectangular ROIs
                    if (roi.regionType == 0 || roi.regionType == 1) {
                        // Convert polygon to rectangle for simple containment check
                        cv::Rect roiRect = cv::boundingRect(roi.polygon);

                        // Check if the vehicle's bounding box intersects with the ROI
                        if ((obj.box & roiRect).area() > 0) {
                            filteredResults.push_back(obj);
                            break;
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

bool Vehicle::HasVehicleCrossedLine(const ANSCENTER::Object& vehicle) {
    if (crossingLineROI.empty()) {
        return false;
    }

    // Get the crossing line
    const auto& line = crossingLineROI[0];
    if (line.polygon.size() < 2) {
        return false;
    }

    // Line is defined by two points
    cv::Point lineStart = line.polygon[0];
    cv::Point lineEnd = line.polygon[1];

    // Get the center bottom point of the vehicle
    cv::Point vehiclePoint(
        vehicle.box.x + vehicle.box.width / 2,
        vehicle.box.y + vehicle.box.height
    );

    // Check if this point is close to the line
    // Calculate distance from point to line
    double lineLength = cv::norm(lineEnd - lineStart);
    double distance = abs((vehiclePoint.y - lineStart.y) * (lineEnd.x - lineStart.x) -
        (vehiclePoint.x - lineStart.x) * (lineEnd.y - lineStart.y)) / lineLength;

    // Consider the vehicle has crossed if it's within a certain threshold distance
    const double THRESHOLD_DISTANCE = 5.0;
    return distance < THRESHOLD_DISTANCE;
}

void Vehicle::UpdateVehicleTracking(const std::vector<ANSCENTER::Object>& vehicles) {
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
                if (!trackedVehicle.crossedLine && HasVehicleCrossedLine(vehicle)) {
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
            newVehicle.crossedLine = HasVehicleCrossedLine(vehicle);
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

int Vehicle::CountVehiclesCrossedLine() {
    int count = 0;
    for (const auto& vehicle : trackedVehicles) {
        if (vehicle.crossedLine) {
            count++;
        }
    }
    return count;
}

bool Vehicle::IsCar(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "car" || vehicle.classId == 0;
}

bool Vehicle::IsTruck(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "truck" || vehicle.classId == 3;
}

bool Vehicle::IsBus(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "bus" || vehicle.classId == 2;
}

bool Vehicle::IsMotorbike(const ANSCENTER::Object& vehicle) {
    return vehicle.className == "motorbike" || vehicle.classId == 1;
}

bool Vehicle::Destroy() {
    // Release resources
    trackedVehicles.clear();
    return true;
}