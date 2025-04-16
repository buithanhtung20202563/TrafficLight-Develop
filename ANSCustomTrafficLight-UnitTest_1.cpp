#include <iostream>
#include <opencv2/opencv.hpp>
#include "CustomLogic.h"

int main() {
    CustomLogic logic;
    std::string modelDir = "C:\\Programs\\DemoAssets\\TrafficLights\\ANS_TrafficLight_v1.0";
    float threshold = 0.5f;
    std::string videoPath = "C:\\Programs\\DemoAssets\\TrafficLights\\trafficlight1.mp4";

    if (!logic.Initialize(modelDir, threshold)) {
        std::cerr << "Init failed\n";
        return -1;
    }
    if (!logic.ConfigureParameters()) {
        std::cerr << "Config failed\n";
        return -1;
    }
    logic.OptimizeModel(true);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video\n";
        return -1;
    }
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        std::vector<ANSCENTER::Object> vehicles, trafficLights;
        logic.RunInference(frame, "cam1", vehicles, trafficLights);
        logic.ProcessViolations(vehicles, trafficLights, "cam1");
        // Optionally visualize
        for (const auto& obj : vehicles) {
            cv::rectangle(frame, obj.box, cv::Scalar(0,255,0), 2);
            cv::putText(frame, obj.className, {obj.box.x, obj.box.y-5}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        }
        for (const auto& obj : trafficLights) {
            cv::rectangle(frame, obj.box, cv::Scalar(0,0,255), 2);
            cv::putText(frame, obj.className, {obj.box.x, obj.box.y-5}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }
        cv::imshow("Result", frame);
        if (cv::waitKey(30) == 27) break;
    }
    logic.Destroy();
    return 0;
}