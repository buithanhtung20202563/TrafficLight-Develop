#include <ANSCustomTrafficLight.h>
#include <iostream>
#include<filesystem>

int main()
{
    std::cout << "Testing custom traffic light!\n";
	ANSCustomTL customTL;
	std::string modelDirectory = "D:\\ANSCustom\\ANS_TrafficLight_v1.0";
	std::string labelMap;
	float detectionScoreThreshold = 0.5f;
	std::string videoFilePath = "D:\\ANSCustom\\ANS_TrafficLight_v1.0\\NgaTu_2025-03-13_03-48-06.mp4";

	// Read the image
    std::string path = "D:\\ANSCustom\\ANSTrafficLight\\ANSCustomTrafficLight\\test2.jpg";

    cv::Mat image = cv::imread(path);
    std::cout << "Version OpenCV: " << CV_VERSION << std::endl;
	// Show the image
    cv::imshow("Image: ", image);
	// Wait for a key press 
    //cv::waitKey(0);

    if (!customTL.Initialize(modelDirectory, detectionScoreThreshold, labelMap)) {
		std::cerr << "Failed to initialize the custom traffic light model.\n";
		return -1;
	}

	// Set parameters for the custom traffic light model
	CustomParams stVehicleParam;
    stVehicleParam.handleId = 0; // Vehicle detector
    stVehicleParam.handleName = "VehicleDetector";
    stVehicleParam.handleParametersJson = {
    {0, "modelType", "1"},                     // int
    {1, "threshold", "0.7"},                   // double
    {4, "objects", "car,truck,bus"}            // list
    };
    
    /* ROI reactangle
    (350, 50) ---- (900, 50)     // pt0 ---- pt1
        |              |
    (350, 250) ---- (900, 250)     // pt3 ---- pt2
    */
    stVehicleParam.ROIs = {
        {0, "DetectArea", { {350, 50}, {900, 50}, {900, 250}, {350, 250} }},
        {2, "CrossingLine", { {100, 100}, {200, 100}}},
        {2, "Direction", { {100, 100}, {200, 100} }}
    };
    CustomParams stTrafficLightParam;
    stTrafficLightParam.handleName = "TrafficLight";
    stTrafficLightParam.handleId = 1;
    stTrafficLightParam.handleParametersJson = {
        {1, "thesHold", "0.5"}
    };
    /* ROI reactangle
     (300, 50) ---- (900, 50)     // pt0 ---- pt1
         |              |
     (300, 100) ---- (900, 100)     // pt3 ---- pt2
     */
    stTrafficLightParam.ROIs = {
        {1, "TrafficRoi", { {300, 50}, {900, 50}, {900, 100}, {300, 100} }}
    };
    std::vector<CustomParams> parameters = { stVehicleParam, stTrafficLightParam };
	customTL.SetParamaters(parameters);

	// Perform inference on the image
    std::vector<CustomObject> detectionResult;
    detectionResult = customTL.RunInference(image, "cameraId");

    for (auto& obj : detectionResult) {
        cv::rectangle(image, obj.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%s:%d", obj.className, obj.classId), cv::Point(obj.box.x, obj.box.y - 5),
            0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    cv::imshow("ANS Object Tracking", image);
    cv::waitKey(0);

    //std::cout << "begin read video" << std::endl;
    //cv::VideoCapture capture(videoFilePath);

    //if (!capture.isOpened()) {
    //    printf("could not read this video file...\n");
    //    return -1;
    //}


    //while (true)
    //{
    //    cv::Mat frame;
    //    if (!capture.read(frame)) // if not success, break loop
    //    {
    //        std::cout << "\n Cannot read the video file. please check your video.\n";
    //        break;
    //    }
    //    std::vector<CustomObject> detectionResult;
    //    detectionResult= customTL.RunInference(frame, "cameraId");

    //    for (auto& obj : detectionResult) {
    //        cv::rectangle(frame, obj.box, cv::Scalar(0, 255, 0), 2);
    //        cv::putText(frame, cv::format("%s:%d", obj.className, obj.classId), cv::Point(obj.box.x, obj.box.y - 5),
    //            0, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    //    }
    //    cv::imshow("ANS Object Tracking", frame);
    //    if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
    //    {
    //        break;
    //    }
    //}
    //capture.release();
    //cv::destroyAllWindows();
    std::cout << "End of program.\n";
    return 0;
}
