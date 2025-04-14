#include "Vehicle.h"
#include <mutex>
#include <chrono>

// Global mutex for thread safety
static std::recursive_mutex g_mutex;

Vehicle::Vehicle() {
    m_modelName = "vehicle";
    m_className = "vehicle.names";
    m_modelType = 4; // TensorRT model by default
    m_detectionType = 1; // Object detection
    m_detectionScoreThreshold = 0.5;
    m_confidenceThreshold = 0.5;
    m_nmsThreshold = 0.5;
}

Vehicle::~Vehicle() {
    Destroy();
}

bool Vehicle::Initialize(const std::string& modelDir, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    
    m_modelDirectory = modelDir;
    m_detectionScoreThreshold = threshold;
    
    // Check engine type and adjust model type if needed
    int engineType = m_detector.GetEngineType();
    if (engineType == 0) {
        // NVIDIA CPU - use ONNX model
        m_modelType = 3;
    }
    
    // Load the vehicle detection model
    std::string licenseKey = "";
    int result = m_detector.LoadModelFromFolder(
        licenseKey.c_str(),
        m_modelName.c_str(),
        m_className.c_str(),
        m_detectionScoreThreshold,
        m_confidenceThreshold,
        m_nmsThreshold,
        1, // Auto detect engine
        m_modelType,
        m_detectionType,
        m_modelDirectory.c_str(),
        m_labelMap
    );
    
    // Configure default parameters
    ConfigureParameters();
    
    return (result == 1);
}

bool Vehicle::Optimize(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (m_detector.Optimize(fp16) == 1);
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
    
    m_detectAreaROI.clear();
    m_detectAreaROI.push_back(detectArea);
    
    // Create the crossing line ROI
    ANSCENTER::Region crossingLine;
    crossingLine.regionType = 2; // Line
    crossingLine.regionName = "CrossingLine";
    
    // Set the line coordinates for the crossing line
    // These coordinates would match the "Crossing Line" shown in the diagram
    crossingLine.polygon.push_back(cv::Point(100, 380));
    crossingLine.polygon.push_back(cv::Point(400, 380));
    
    m_crossingLineROI.clear();
    m_crossingLineROI.push_back(crossingLine);
    
    // Create the direction line ROI
    ANSCENTER::Region directionLine;
    directionLine.regionType = 4; // Direction line
    directionLine.regionName = "Direction";
    
    // Set the line coordinates for the direction line
    directionLine.polygon.push_back(cv::Point(250, 350));
    directionLine.polygon.push_back(cv::Point(250, 250));
    
    m_directionLineROI.clear();
    m_directionLineROI.push_back(directionLine);
    
    // Create parameter structure
    ANSCENTER::Params param;
    param.handleId = 0; // Vehicle detector ID
    param.handleName = m_modelName;
    
    // Add threshold parameter
    ANSCENTER::ParamType thresholdParam;
    thresholdParam.type = 1; // double
    thresholdParam.name = "threshold";
    thresholdParam.value = std::to_string(m_detectionScoreThreshold);
    
    param.handleParametersJson.push_back(thresholdParam);
    
    // Add all ROIs to the parameter
    param.ROIs.insert(param.ROIs.end(), m_detectAreaROI.begin(), m_detectAreaROI.end());
    param.ROIs.insert(param.ROIs.end(), m_crossingLineROI.begin(), m_crossingLineROI.end());
    param.ROIs.insert(param