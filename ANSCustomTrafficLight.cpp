#include "ANSCustomTrafficLight.h"
// #define FNS_DEBUG
ANSCustomTL::ANSCustomTL()
{
	// Initialize the model
}
bool ANSCustomTL::OptimizeModel(bool fp16)
{
	bool result = false;
	bool vResult = vehicleDetector.Optimize(fp16);
	bool tResult = trafficLightDetector.Optimize(fp16);
	return vResult && tResult;
}
std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat &input)
{
	return RunInference(input, "CustomCam");
}
bool ANSCustomTL::Destroy()
{
	// Both detectors are released here
	return true;
}
bool ANSCustomTL::Initialize(const std::string &modelDirectory, float detectionScoreThreshold, std::string &labelMap)
{
	// Set parameters for the custom traffic light model
	CustomParams stVehicleParam;
	stVehicleParam.handleId = 0; // Vehicle detector
	stVehicleParam.handleName = "VehicleDetector";
	stVehicleParam.handleParametersJson = {
		{0, "modelType", "1"},			// int
		{1, "threshold", "0.7"},		// double
		{4, "objects", "car,truck,bus"} // list
	};

	/* ROI reactangle
	(250, 50) ---- (900, 50)     // pt0 ---- pt1
		|              |
	(250, 35280) ---- (900, 280)     // pt3 ---- pt2
	*/
	stVehicleParam.ROIs = {
		{0, "DetectArea", {{250, 50}, {900, 50}, {900, 280}, {250, 280}}},
		{1, "CrossingLine", {{900, 280}, {250, 280}}},
		{2, "Direction", {{100, 100}, {200, 100}}}};
	CustomParams stTrafficLightParam;
	stTrafficLightParam.handleName = "TrafficLight";
	stTrafficLightParam.handleId = 1;
	stTrafficLightParam.handleParametersJson = {
		{1, "thesHold", "0.5"}};
	/* ROI reactangle
	 (300, 50) ---- (900, 50)     // pt0 ---- pt1
		 |              |
	 (300, 100) ---- (900, 100)     // pt3 ---- pt2
	 */
	stTrafficLightParam.ROIs = {
		{1, "TrafficRoi", {{300, 50}, {900, 50}, {900, 100}, {300, 100}}}};
	std::vector<CustomParams> parameters = {stVehicleParam, stTrafficLightParam};
	this->SetParamaters(parameters);
	// 1. The modelDirectory is supplied by ANSVIS and contains the path to the model files
	_modelDirectory = modelDirectory;
	_detectionScoreThreshold = detectionScoreThreshold;

	// 2. User can start impelementing the initialization logic here
	bool vehicleResult = m_cVehicleDetector.Initialize(_modelDirectory, _detectionScoreThreshold);
	bool lightResult = m_cTrafficLightDetector.Initialize(_modelDirectory, _detectionScoreThreshold);

	// 3 Create label map
	// Based on two class names, please form the label map (e.g. vehicle.names and light.names)
	// We stack the two class names together to form the label map
	labelMap = "car,motorbike,bus,truck,bike,container,tricycle,human,green,red,yellow";
	if ((vehicleResult == 1) && (lightResult == 1))
		return true;
	return false;
}
ANSCustomTL::~ANSCustomTL()
{
	Destroy();
}
bool ANSCustomTL::ConfigureParamaters(std::vector<CustomParams> &param)
{
	if (this->_param.size() == 0)
	{
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
	else
	{
		for (const auto &p : this->_param)
		{
			param.push_back(p);
		}
	}
	return true;
}

cv::Mat cropFromFourPoints(const cv::Mat &src, const std::vector<cv::Point> &pts)
{
	if (pts.size() != 4)
		throw std::runtime_error("Need 4 points");

	std::vector<cv::Point2f> ordered = {
		pts[0], pts[1], pts[2], pts[3]};

	float width = max(cv::norm(ordered[0] - ordered[1]),
					  cv::norm(ordered[2] - ordered[3]));
	float height = max(cv::norm(ordered[0] - ordered[3]),
					   cv::norm(ordered[1] - ordered[2]));

	std::vector<cv::Point2f> dst = {
		{0.0f, 0.0f},
		{width - 1, 0.0f},
		{width - 1, height - 1},
		{0.0f, height - 1}};

	cv::Mat M = cv::getPerspectiveTransform(ordered, dst);
	cv::Mat cropped;
	cv::warpPerspective(src, cropped, M, cv::Size(width, height));
	return cropped;
}

std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat &input, const std::string &camera_id)
{

	std::lock_guard<std::recursive_mutex> lock(_mutex);
	std::vector<CustomObject> results;
	try
	{
		std::vector<ANSCENTER::Object> outputVehicle;
		std::vector<ANSCENTER::Object> outputTrafficLight;
		CustomParams stVehicleParam = m_cVehicleDetector.GetParameters();
		std::vector<cv::Point> vDetectArea{};
		std::vector<cv::Point> vCrossLine{};
		std::vector<cv::Point> vDirection{};

		// Extract ROIs from parameters
		for (const auto &roi : stVehicleParam.ROIs)
		{
			if (roi.regionName == "DetectArea")
			{
				vDetectArea = roi.polygon;
			}
			else if (roi.regionName == "CrossingLine")
			{
				vCrossLine = roi.polygon;
			}
			else if (roi.regionName == "Direction")
			{
				vDirection = roi.polygon;
			}
		}

		// Create ROI mask for vehicle detection
		cv::Mat roiMask = cv::Mat::zeros(input.size(), CV_8UC1);
		if (!vDetectArea.empty())
		{
			std::vector<std::vector<cv::Point>> contours = {vDetectArea};
			cv::fillPoly(roiMask, contours, cv::Scalar(255));
		}

		// Run vehicle detection only within ROI
		std::vector<ANSCENTER::Object> vOutVehicle = m_cVehicleDetector.DetectVehicles(input, "cameraID");

		// Filter vehicles to only those within the detection area
		std::vector<ANSCENTER::Object> filteredVehicles;
		for (const auto &vehicle : vOutVehicle)
		{
			cv::Point center(vehicle.box.x + vehicle.box.width / 2,
							 vehicle.box.y + vehicle.box.height / 2);
			if (cv::pointPolygonTest(vDetectArea, center, false) >= 0)
			{
				filteredVehicles.push_back(vehicle);
			}
		}

		// Run traffic light detection
		CustomParams stTrafficParam = m_cTrafficLightDetector.GetParameters();
		std::vector<cv::Point> vTrafficArea{};

		for (const auto &roi : stTrafficParam.ROIs)
		{
			if (roi.regionName == "TrafficRoi")
			{
				vTrafficArea = roi.polygon;
			}
		}

		// Create ROI mask for traffic light detection
		cv::Mat trafficRoiMask = cv::Mat::zeros(input.size(), CV_8UC1);
		if (!vTrafficArea.empty())
		{
			std::vector<std::vector<cv::Point>> contours = {vTrafficArea};
			cv::fillPoly(trafficRoiMask, contours, cv::Scalar(255));
		}

		cv::Mat cvTrafficImg = cropFromFourPoints(input, vTrafficArea);
		std::vector<ANSCENTER::Object> vOutTrafficLight = m_cTrafficLightDetector.DetectTrafficLights(cvTrafficImg, "cameraID");

		// Combine the results
		std::map<std::string, int> vehicleTypeTrackIds; // Map to store track IDs for each vehicle type
		for (const auto &obj : filteredVehicles)
		{
			CustomObject customObj;
			customObj.classId = obj.classId;
			// Get or initialize track ID for this vehicle type
			if (vehicleTypeTrackIds.find(obj.className) == vehicleTypeTrackIds.end()) {
				vehicleTypeTrackIds[obj.className] = 0;
			}
			customObj.trackId = vehicleTypeTrackIds[obj.className]++;
			customObj.className = obj.className;
			customObj.confidence = obj.confidence;
			customObj.box = obj.box;
			customObj.cameraId = camera_id;
			results.push_back(customObj);
		}

		// Traffic light detection
		std::map<std::string, int> trafficLightTypeTrackIds; // Map to store track IDs for each traffic light type
		for (const auto &obj : vOutTrafficLight)
		{
			CustomObject customObj;
			customObj.classId = obj.classId;
			// Get or initialize track ID for this traffic light type
			if (trafficLightTypeTrackIds.find(obj.className) == trafficLightTypeTrackIds.end()) {
				trafficLightTypeTrackIds[obj.className] = 0;
			}
			customObj.trackId = trafficLightTypeTrackIds[obj.className]++;
			customObj.className = obj.className;
			customObj.confidence = obj.confidence;
			customObj.box = obj.box;
			customObj.cameraId = camera_id;
			results.push_back(customObj);
		}

		// Draw ROIs on the image for visualization
		if (!vDetectArea.empty())
		{
			std::vector<std::vector<cv::Point>> contours = {vDetectArea};
			cv::polylines(input, contours, true, cv::Scalar(0, 255, 0), 2);
			cv::putText(input, "Detection Area", vDetectArea[0],
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
		}

		if (!vCrossLine.empty())
		{
			cv::line(input, vCrossLine[0], vCrossLine[1], cv::Scalar(0, 0, 255), 2);
			cv::putText(input, "Crossing Line", vCrossLine[0],
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}

		if (!vTrafficArea.empty())
		{
			std::vector<std::vector<cv::Point>> contours = {vTrafficArea};
			cv::polylines(input, contours, true, cv::Scalar(255, 0, 0), 2);
			cv::putText(input, "Traffic Light Area", vTrafficArea[0],
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
		}

		// Draw detected vehicles with their information
		for (const auto &vehicle : filteredVehicles)
		{
			// Draw bounding box
			cv::Scalar boxColor(0, 255, 0); // Green color for normal vehicles
			if (m_cVehicleDetector.IsVehicleCrossedLine(vehicle))
			{
				boxColor = cv::Scalar(0, 0, 255); // Red color for violating vehicles
			}
			cv::rectangle(input, vehicle.box, boxColor, 2);

			// Prepare vehicle information text
			std::string vehicleInfo = cv::format("%s (ID:%d) %.2f",
												 vehicle.className.c_str(),
												 vehicle.trackId,
												 vehicle.confidence);

			// Calculate text position
			cv::Point textPos(vehicle.box.x, vehicle.box.y - 10);
			if (textPos.y < 20)
				textPos.y = vehicle.box.y + 20; // Adjust if text would go above image

			// Draw background rectangle for text
			int baseline = 0;
			cv::Size textSize = cv::getTextSize(vehicleInfo, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
			cv::rectangle(input,
						  cv::Point(textPos.x - 2, textPos.y - textSize.height - 2),
						  cv::Point(textPos.x + textSize.width + 2, textPos.y + 2),
						  cv::Scalar(0, 0, 0), cv::FILLED);

			// Draw vehicle information text
			cv::putText(input, vehicleInfo, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

			// Draw vehicle center point
			cv::Point center(vehicle.box.x + vehicle.box.width / 2,
							 vehicle.box.y + vehicle.box.height / 2);
			cv::circle(input, center, 3, boxColor, -1);
		}

		// Draw traffic lights
		for (auto &obj : results)
		{
			if (obj.className == "red" || obj.className == "green" || obj.className == "yellow")
			{
				obj.box.x += vTrafficArea[0].x;
				obj.box.y += vTrafficArea[0].y;
				cv::rectangle(input, obj.box, cv::Scalar(0, 255, 0), 2); // Vẽ khung màu xanh cho đèn tín hiệu

				// Chuẩn bị vẽ chữ trắng trên nền đen
				std::string text = cv::format("%s ID:%d %.2f", obj.className, obj.classId, obj.confidence);
				cv::Point textPos(obj.box.x, obj.box.y - 5);

				// Tính toán kích thước văn bản và vẽ nền đen
				int baseline = 0;
				cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
				cv::rectangle(input,
							  cv::Point(textPos.x - 2, textPos.y - textSize.height - 2),
							  cv::Point(textPos.x + textSize.width + 2, textPos.y + 2),
							  cv::Scalar(0, 0, 0), cv::FILLED);

				// Vẽ chữ trắng
				cv::putText(input, text, textPos,
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			}
		}

		// Check if traffic light is red
		bool isRedLight = false;
		std::cout << "\n============= Traffic Light Analysis =============" << std::endl;
		std::cout << "Camera ID: " << camera_id << std::endl;
		std::cout << "Number of traffic lights detected: " << vOutTrafficLight.size() << std::endl;

		for (const auto &obj : vOutTrafficLight)
		{
			std::cout << "Traffic Light - Class: " << obj.className
					  << ", ID: " << obj.classId
					  << ", Confidence: " << obj.confidence
					  << ", Position: (" << obj.box.x << "," << obj.box.y << ")" << std::endl;

			if (obj.className == "red" && obj.confidence > 0.5)
			{
				isRedLight = true;
				std::cout << "RED LIGHT STATE CONFIRMED - Monitoring for violations" << std::endl;
				break;
			}
		}

		// If red light is detected, check for vehicles in the detection area
		if (isRedLight)
		{
			std::cout << "\n============= Vehicle Detection Analysis =============" << std::endl;
			std::cout << "Total vehicles detected: " << filteredVehicles.size() << std::endl;

			for (const auto &vehicle : filteredVehicles)
			{
				std::cout << "\n----- Vehicle Details -----" << std::endl;
				std::cout << "Type: " << vehicle.className << std::endl;
				std::cout << "Track ID: " << vehicle.trackId << std::endl;
				std::cout << "Confidence: " << vehicle.confidence << std::endl;
				std::cout << "Position: (" << vehicle.box.x << "," << vehicle.box.y << ")" << std::endl;
				std::cout << "Size: " << vehicle.box.width << "x" << vehicle.box.height << std::endl;

				// Kiểm tra xem phương tiện có nằm trong vùng detect không
				cv::Point center(vehicle.box.x + vehicle.box.width / 2,
								 vehicle.box.y + vehicle.box.height / 2);
				bool isInDetectArea = cv::pointPolygonTest(vDetectArea, center, false) >= 0;

				if (isInDetectArea)
				{
					std::cout << "Vehicle is inside detection area" << std::endl;

					// Kiểm tra vi phạm
					bool isViolation = m_cVehicleDetector.IsVehicleCrossedLine(vehicle);

					if (isViolation)
					{
						std::cout << "\n!!! RED LIGHT VIOLATION DETECTED !!!" << std::endl;
						std::cout << "Location: Camera " << camera_id << std::endl;
						std::cout << "Violation Details:" << std::endl;
						std::cout << "- Vehicle Type: " << vehicle.className << std::endl;
						std::cout << "- Track ID: " << vehicle.trackId << std::endl;
						std::cout << "- Detection Confidence: " << vehicle.confidence << std::endl;
						std::cout << "- Vehicle Position: (" << vehicle.box.x << "," << vehicle.box.y << ")" << std::endl;
						std::cout << "- Vehicle Size: " << vehicle.box.width << "x" << vehicle.box.height << std::endl;
						std::cout << "=========================================" << std::endl;

						// Vẽ thông báo vi phạm
						cv::Scalar violationColor(0, 0, 255); // Red color
						cv::rectangle(input, vehicle.box, violationColor, 3);

						std::string violationText = "VIOLATION #" + std::to_string(vehicle.trackId);
						cv::Point textPos(vehicle.box.x, vehicle.box.y - 10);

						// Vẽ nền cho text
						cv::Size textSize = cv::getTextSize(violationText, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
						cv::rectangle(input,
									  cv::Point(textPos.x - 5, textPos.y - textSize.height - 5),
									  cv::Point(textPos.x + textSize.width + 5, textPos.y + 5),
									  violationColor, cv::FILLED);

						cv::putText(input, violationText, textPos,
									cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
					}
					else
					{
						std::cout << "Vehicle is in detection area but has not crossed the line" << std::endl;
					}
				}
				else
				{
					std::cout << "Vehicle is outside detection area" << std::endl;
				}
			}
		}
		else
		{
			std::cout << "\nNo red light detected - Traffic flowing normally" << std::endl;
			std::cout << "Total vehicles in frame: " << filteredVehicles.size() << std::endl;
		}

		return results;
	}

	catch (std::exception &e)
	{
		return results;
	}
}

bool ANSCustomTL::SetParamaters(const std::vector<CustomParams> &param)
{
	_param.clear();
	for (const auto &p : param)
	{
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