
#include "TrafficLight_Test.h"

void TrafficLightTest() {
    // Initialize the custom logic
    CustomLogic logic;
    logic.Initialize("/path/to/models", 0.5);
    logic.OptimizeModel(true); // Use FP16 optimization
    
    // Set up violation callback
    logic.SetViolationCallback([](const std::string& message, const cv::Mat& frame, const cv::Rect& box) {
        // Save evidence image
        cv::Mat evidence = frame.clone();
        cv::rectangle(evidence, box, cv::Scalar(0, 0, 255), 2);
        cv::imwrite("violation_" + std::to_string(time(nullptr)) + ".jpg", evidence);
        
        // Notify operator
        std::cout << "ALERT: " << message << std::endl;
    });
    
    // Process frames from camera
    cv::VideoCapture cap(0);
    cv::Mat frame;
    
    while (cap.read(frame)) {
        logic.ProcessFrame(frame, "Camera1");
    }
    
}