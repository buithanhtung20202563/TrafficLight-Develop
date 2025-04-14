#include "ANSCustomTrafficLight.h"
//#define FNS_DEBUG
ANSCustomTL::ANSCustomTL()
{
	// Initialize the model
}
bool ANSCustomTL::OptimizeModel(bool fp16)
{
	bool result = false;
	bool vResult= vehicleDetector.Optimize(fp16);
	bool tResult = trafficLightDetector.Optimize(fp16);
	return vResult && tResult;
}
std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat& input)
{
	return RunInference(input, "CustomCam");
}
bool ANSCustomTL::Destroy()
{
	// Both detectors are released here
	return true;
}
bool ANSCustomTL::Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap)
{
	//1. The modelDirectory is supplied by ANSVIS and contains the path to the model files
	_modelDirectory = modelDirectory;
	_detectionScoreThreshold = detectionScoreThreshold;

	_vehicleModelName = "vehicle";
	_trafficLightModelName = "light";

	_vehicleClassName ="vehicle.names";
	_trafficLightClassName = "light.names";
	_vehicleModelType = 4;				// Assuming to use TensorRT model. Please refer to ANSLIB.h for more model types
	_vehicleDetectionType = 1;			// This is object detection type (so the type is 1)

	_trafficLightModelType = 4;			// Assuming to use TensorRT model. Please refer to ANSLIB.h for more model types
	_trafficLightDetectionType = 1;		// This is object detection type (so the type is 1)
	int engineType = vehicleDetector.GetEngineType();
	if (engineType == 0) {// NVIDIA CPU
		_vehicleModelType = 3;				// Assuming to use Yolo onnx model. Please refer to ANSLIB.h for more model types
		_vehicleDetectionType = 1;			// This is object detection type (so the type is 1)

		_trafficLightModelType = 3;			// Assuming to use Yolo onnx model. Please refer to ANSLIB.h for more model types
		_trafficLightDetectionType = 1;		// This is object detection type (so the type is 1)
	}



	//2. User can start impelementing the initialization logic here
	double _vehicleModelConfThreshold = 0.5;
	double _vehicleModelNMSThreshold = 0.5;
	std::string licenseKey = "";
	int vehicleResult= vehicleDetector.LoadModelFromFolder(licenseKey.c_str(),
														  _vehicleModelName.c_str(),
														 _vehicleClassName.c_str(),
														  _detectionScoreThreshold, 
														  _vehicleModelConfThreshold, 
														  _vehicleModelNMSThreshold,1, 
														  _vehicleModelType, 
														  _trafficLightDetectionType,
														  _modelDirectory.c_str(), 
														  _vehicleLabelMap);


	double _trafficLightModelConfThreshold = 0.5;
	double _trafficLightModelNMSThreshold = 0.5;
	int lightResult = trafficLightDetector.LoadModelFromFolder(licenseKey.c_str(),
										_trafficLightModelName.c_str(),
										_trafficLightClassName.c_str(),
										_detectionScoreThreshold,
										_trafficLightModelConfThreshold,
										_trafficLightModelNMSThreshold, 1,
										_trafficLightModelType,
										_trafficLightDetectionType,
										_modelDirectory.c_str(),
										_trafficLightLabelMap);
	// 3 Create label map
	// Based on two class names, please form the label map (e.g. vehicle.names and light.names)
	// We stack the two class names together to form the label map
	labelMap = "car,motorbike,bus,truck,bike,container,tricycle,human,green,red,yellow";
	if ((vehicleResult == 1) && (lightResult == 1)) return true;
	return false;
}
ANSCustomTL::~ANSCustomTL()
{
	Destroy();
}
bool ANSCustomTL::ConfigureParamaters(std::vector<CustomParams>& param)
{
	if (this->_param.size() == 0) {
		// We need to create parameter schema here
		CustomParams p;
		p.handleId = 0; // Vehicle detector
		p.handleName = _vehicleModelName;

		// Define parameters 
		std::vector<CustomParamType> handleParametersJson;
		CustomParamType param1;
		param1.type = 0; // int
		param1.name = "modelType";
		param1.value = _detectionScoreThreshold;
		handleParametersJson.push_back(param1);
		// Add more parameters as needed
		p.handleParametersJson = handleParametersJson;
		// Define ROIs
		std::vector<CustomRegion> ROIs;
		CustomRegion roi;
		roi.regionType = 0; // Rectangle
		roi.regionName = "DetectArea";
		roi.polygon.push_back(cv::Point(0, 0));
		roi.polygon.push_back(cv::Point(100, 0));
		roi.polygon.push_back(cv::Point(0, 100));
		roi.polygon.push_back(cv::Point(100, 100));
		// Add more points to the polygon as needed
		ROIs.push_back(roi);
		p.ROIs = ROIs;
		param.push_back(p);


		CustomParams p1;
		p1.handleId = 2; // Light detector
		p1.handleName = _trafficLightModelName;

		// Define parameters 
		std::vector<CustomParamType> handleParametersJson1;
		CustomParamType param2;
		param2.type = 1; // double
		param2.name = "thesHold";
		param2.value = _detectionScoreThreshold;
		handleParametersJson1.push_back(param2);
		// Add more parameters as needed
		p1.handleParametersJson = handleParametersJson1;
		// Define ROIs
		std::vector<CustomRegion> ROIs1;
		CustomRegion roi1;
		roi1.regionType = 0; // Rectangle
		roi1.regionName = "TrafficRoi";
		roi1.polygon.push_back(cv::Point(0, 0));
		roi1.polygon.push_back(cv::Point(100, 0));
		roi1.polygon.push_back(cv::Point(0, 100));
		roi1.polygon.push_back(cv::Point(100, 100));
		ROIs1.push_back(roi1);
		p1.ROIs = ROIs1;
		param.push_back(p);
	}
	else {
		for (const auto& p : this->_param) {
			param.push_back(p);
		}
	}
	return true;
}
std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat& input, const std::string& camera_id) {
	std::lock_guard<std::recursive_mutex> lock(_mutex);
	std::vector<CustomObject> results;
	try {

		// We can access to the parameters here
		int vehicleDetectorId = this->_param[0].handleId;  /// Vehicle detector
		std::string handleName = this->_param[0].handleName;
		std::vector<cv::Point> directionLine = this->_param[0].ROIs[2].polygon;
		std::vector<CustomParamType> handleParametersJson = this->_param[0].handleParametersJson;

		int trafficLightId = this->_param[1].handleId;  /// Traffic light detector

		// Then we can do customization based on the parameters

		std::vector<ANSCENTER::Object> outputVehicle;
		std::vector<ANSCENTER::Object> outputTrafficLight;
		// Run vehicle detection
		vehicleDetector.RunInference(input, camera_id.c_str(), outputVehicle);
		// Run traffic light detection
		trafficLightDetector.RunInference(input, camera_id.c_str(), outputTrafficLight);
		// Combine the results
		for (const auto& obj : outputVehicle) {
			CustomObject customObj;
			customObj.classId = obj.classId;
			customObj.trackId = obj.trackId;
			customObj.className = obj.className;
			customObj.confidence = obj.confidence;
			customObj.box = obj.box;
			customObj.cameraId = camera_id;
			results.push_back(customObj);
		}

		// Traffic light detection class IDs are 8, 9, 10
		// Because we stack the two class names together, we need to adjust the class IDs
		int maxVehicleClassId = 7;
		for (const auto& obj : outputTrafficLight) {
			CustomObject customObj;
			customObj.classId = obj.classId+ maxVehicleClassId;
			customObj.trackId = obj.trackId;
			customObj.className = obj.className;
			customObj.confidence = obj.confidence;
			customObj.box = obj.box;
			customObj.cameraId = camera_id;
			results.push_back(customObj);
		}
		// Add additional information if needed
		return results;
	}
	catch (std::exception& e) {
		return results;
	}
}

