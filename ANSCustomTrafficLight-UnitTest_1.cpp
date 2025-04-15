#include "CustomLogic.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "Testing custom traffic logic!\n";
    CustomLogic customLogic;
    std::string modelDirectory = "C:\\Programs\\DemoAssets\\TrafficLights\\ANS_TrafficLight_v1.0";
    std::string labelMap;
    float detectionScoreThreshold = 0.5f;
    std::string videoFilePath = "C:\\Programs\\DemoAssets\\TrafficLights\\trafficlight1.mp4";

    // Initialize the custom logic
    if (!customLogic.Initialize(modelDirectory, detectionScoreThreshold, labelMap)) {
        std::cerr << "Failed to initialize the custom traffic logic model.\n";
        return -1;
    }

    // Configure parameters
    std::vector<CustomParams> params;
    if (!customLogic.ConfigureParamaters(params)) {
        std::cerr << "Failed to configure parameters.\n";
        return -1;
    }

    // Optimize model (optional)
    if (!customLogic.OptimizeModel(true)) {
        std::cerr << "Failed to optimize model (continuing anyway).\n";
    }

    std::cout << "Begin reading video" << std::endl;
    cv::VideoCapture capture(videoFilePath);

    if (!capture.isOpened()) {
        std::cerr << "Could not read the video file: " << videoFilePath << "\n";
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        if (!capture.read(frame)) // If not successful, break loop
        {
            std::cout << "\nCannot read the video file. Please check your video.\n";
            break;
        }

        // Run inference (detects vehicles and traffic lights, logs violations)
        std::vector<CustomObject> detectionResult = customLogic.RunInference(frame, "cameraId");

        // Visualize detections
        for (const auto& obj : detectionResult) {
            // Use different colors for vehicles (green) and traffic lights (blue)
            cv::Scalar color = (obj.classId >= 8) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, obj.box, color, 2);
            cv::putText(frame,
                cv::format("%s:%d", obj.className.c_str(), obj.classId),
                cv::Point(obj.box.x, obj.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        cv::imshow("Traffic Violation Detection", frame);
        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    customLogic.Destroy();
    std::cout << "End of program.\n";
    return 0;
}