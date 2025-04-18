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

	//2. User can start impelementing the initialization logic here
	bool vehicleResult = m_cVehicleDetector.Initialize(_modelDirectory, _detectionScoreThreshold);
	bool lightResult = m_cTrafficLightDetector.Initialize(_modelDirectory, _detectionScoreThreshold);
	
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

cv::Mat cropFromFourPoints(const cv::Mat& src, const std::vector<cv::Point>& pts) {
	if (pts.size() != 4) throw std::runtime_error("Need 4 points");

	std::vector<cv::Point2f> ordered = {
		pts[0], pts[1], pts[2], pts[3]
	};

	float width = max(cv::norm(ordered[0] - ordered[1]),
		cv::norm(ordered[2] - ordered[3]));
	float height = max(cv::norm(ordered[0] - ordered[3]),
		cv::norm(ordered[1] - ordered[2]));

	std::vector<cv::Point2f> dst = {
		{0.0f, 0.0f},
		{width - 1, 0.0f},
		{width - 1, height - 1},
		{0.0f, height - 1}
	};

	cv::Mat M = cv::getPerspectiveTransform(ordered, dst);
	cv::Mat cropped;
	cv::warpPerspective(src, cropped, M, cv::Size(width, height));
	return cropped;
}

std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat& input, const std::string& camera_id) {
	std::lock_guard<std::recursive_mutex> lock(_mutex);
	std::vector<CustomObject> results;
	try {
		std::vector<ANSCENTER::Object> outputVehicle;
		std::vector<ANSCENTER::Object> outputTrafficLight;
		CustomParams stVehicleParam = m_cVehicleDetector.GetParameters();
		std::vector<cv::Point> vDetectArea{};
		std::vector<cv::Point> vCrossLine{};
		std::vector<cv::Point> vDirection{};
		for (const auto& roi : stVehicleParam.ROIs) {
			if (roi.regionName == "DetectArea") {
				vDetectArea = roi.polygon;
			}
			else if (roi.regionName == "CrossingLine") {
				// Do something with crossing line
				vCrossLine = roi.polygon;
			}
			else {
				// Do something with direction line
				vDirection = roi.polygon;
			}
		}

		CustomParams stTrafficParam = m_cTrafficLightDetector.GetParameters();
		std::vector<cv::Point> vTrafficArea{};

		for (const auto& roi : stTrafficParam.ROIs) {
			if (roi.regionName == "TrafficRoi") {
				vTrafficArea = roi.polygon;
			}
		}

		cv::Mat cvTrafficImg = cropFromFourPoints(input, vTrafficArea);
		cv::Mat cvVehicleImg = cropFromFourPoints(input, vDetectArea);

		cv::imshow("Crop Image", cvTrafficImg);
		cv::imshow("Crop Image", cvVehicleImg);
		//cv::waitKey(0);
		// Run vehicle detection
		std::vector<ANSCENTER::Object> vOutVehicle = m_cVehicleDetector.DetectVehicles(input, "cameraID");

		// Run traffic light detection
		std::vector<ANSCENTER::Object> vOutTrafficLight = m_cTrafficLightDetector.DetectTrafficLights(cvTrafficImg, "cameraID");
		// TungBT: Modify logic passing red light
		// Combine the results
		for (const auto& obj : vOutVehicle) {
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
		for (const auto& obj : vOutTrafficLight) {
			CustomObject customObj;
			customObj.classId = obj.classId + maxVehicleClassId;
			customObj.trackId = obj.trackId;
			customObj.className = obj.className;
			customObj.confidence = obj.confidence;
			customObj.box = obj.box;
			customObj.cameraId = camera_id;
			results.push_back(customObj);
		}
		for (auto& obj : results) {
			if (obj.className == "red" || obj.className == "green" || obj.className == "yellow")
			{
				obj.box.x += vTrafficArea[0].x;
				obj.box.y += vTrafficArea[0].y;
				cv::rectangle(input, obj.box, cv::Scalar(0, 255, 0), 2);
				cv::putText(input, cv::format("%s:%d", obj.className, obj.classId), cv::Point(obj.box.x, obj.box.y - 5),
					0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			}
			else
			{
				//obj.box.x += vVehicleArea[0].x;
				//obj.box.y += vVehicleArea[0].y;
				cv::rectangle(input, obj.box, cv::Scalar(0, 255, 0), 2);
				cv::putText(input, cv::format("%s:%d", obj.className, obj.classId), cv::Point(obj.box.x, obj.box.y - 5),
					0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			}
		}
		// Check if traffic light is red
		bool isRedLight = false;
		for (const auto& obj : vOutTrafficLight) {
			if (obj.className == "red" || obj.classId == 9) { // Check both className and classId
				isRedLight = true;
				std::cout << "RED LIGHT DETECTED!" << std::endl;
				break;
			}
		}

		// If red light is detected, check for vehicles in the detection area
		if (isRedLight) {
			std::cout << "Checking vehicles... Found: " << vOutVehicle.size() << " vehicles" << std::endl;
			
			for (const auto& vehicle : vOutVehicle) {
				// Debug print vehicle info
				std::cout << "Checking vehicle: " << vehicle.className 
					<< " at position: (" << vehicle.box.x << "," << vehicle.box.y << ")" << std::endl;
				
				// Check if vehicle is in detection area
				if (m_cVehicleDetector.IsVehicleCrossedLine(vehicle)) {
					// Log vehicle information with timestamp
					time_t now = time(0);
					char* dt = ctime(&now);
					std::cout << "\n!!! RED LIGHT VIOLATION DETECTED !!!" << std::endl;
					std::cout << "Time: " << dt;
					std::cout << "Vehicle Type: " << vehicle.className << std::endl;
					std::cout << "Track ID: " << vehicle.trackId << std::endl;
					std::cout << "Confidence: " << vehicle.confidence << std::endl;
					std::cout << "Position: (" << vehicle.box.x << ", " << vehicle.box.y << ")" << std::endl;
					std::cout << "Size: " << vehicle.box.width << "x" << vehicle.box.height << std::endl;
					std::cout << "----------------------------------------" << std::endl;

					// Draw violation box on the image
					cv::rectangle(input, vehicle.box, cv::Scalar(0, 0, 255), 3); // Red box for violating vehicle
					cv::putText(input, "VIOLATION", 
						cv::Point(vehicle.box.x, vehicle.box.y - 10),
						cv::FONT_HERSHEY_SIMPLEX, 0.8, 
						cv::Scalar(0, 0, 255), 2);
				}
			}
		} else {
			std::cout << "No red light detected" << std::endl;
		}

		cv::imshow("ANS Object Tracking", input);
		cv::waitKey(1); // Changed from waitKey(0) to waitKey(1) for continuous processing
		return results;
	}

	catch (std::exception& e) {
		return results;
	}
}

bool ANSCustomTL::SetParamaters(const std::vector<CustomParams>& param)
{
	_param.clear();
	for (const auto& p : param) {
		_param.push_back(p);
		if (p.handleName == "VehicleDetector" || p.handleId == 0) 
		{
			// Set parameters for vehicle detector
			m_cVehicleDetector.SetParameters(p);
		}
		else
		{
			m_cTrafficLightDetector.SetParameters(p);
		}
	}
	return true;
}
