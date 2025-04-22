#include <ANSCustomTrafficLight.h>
#include <iostream>
#include <filesystem>

int main()
{
    std::cout << "Testing custom traffic light!\n";
    ANSCustomTL customTL;

    // Chuyển đổi các đường dẫn tuyệt đối thành đường dẫn tương đối
    std::string modelDirectory = "..\\ANS_TrafficLight_v1.0";
    std::string labelMap;
    float detectionScoreThreshold = 0.5f;
    std::string videoFilePath = "..\\NgaTu_2025-03-13_03-48-06.mp4";

    // Đọc ảnh từ đường dẫn tương đối
    std::string path = "..\\test_4.jpg";

    cv::Mat image = cv::imread(path);
    std::cout << "Version OpenCV: " << CV_VERSION << std::endl;

    if (!customTL.Initialize(modelDirectory, detectionScoreThreshold, labelMap))
    {
        std::cerr << "Failed to initialize the custom traffic light model.\n";
        return -1;
    }
    // Perform inference on the image
    std::vector<CustomObject> detectionResult;
    detectionResult = customTL.RunInference(image, "cameraId");

    // Lưu ảnh đầu ra vào đường dẫn tương đối
    std::string outputImagePath = "..\\ANS_Object_Tracking_Result.jpg";
    cv::imwrite(outputImagePath, image);

    cv::imshow("ANS Object Tracking", image);
    cv::waitKey(0);

    std::cout << "End of program.\n";
    return 0;
}