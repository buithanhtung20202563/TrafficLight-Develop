
#include "TrafficLight_Test.h"

    void TrafficLightTest() {
    // Tạo instance
    CustomLogic detector;

    // Khởi tạo
    std::string labelMap;
    detector.Initialize("/path/to/models", 0.5, labelMap);

    // Đặt callback cho vi phạm
    detector.SetViolationCallback([](const std::string& cameraId, const CustomObject& vehicle) {
        std::cout << "Vi phạm phát hiện tại camera " << cameraId
                << ", phương tiện loại: " << vehicle.className
                << ", ID: " << vehicle.trackId << std::endl;
    });

    // Chạy phát hiện
    cv::Mat frame = cv::imread("frame.jpg");
    std::vector<CustomObject> results = detector.RunInference(frame, "Camera1");
}