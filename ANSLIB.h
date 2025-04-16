#ifndef ANSLIB_H
#define ANSLIB_H
#define ANSLIB_API __declspec(dllexport)
#pragma once
#include <windows.h>
#include <opencv2/opencv.hpp>
enum DetectionType {
	CLASSIFICATION = 0,
	DETECTION = 1,
	SEGMENTATION = 2,
	FACEDETECTOR = 3,
	FACERECOGNIZER = 4,
	LICENSEPLATE = 5,
	TEXTSCENSE = 6,
	KEYPOINT = 7
};
enum ModelType {
	TENSORFLOW = 0,
	YOLOV4 = 1,
	YOLOV5 = 2,
	YOLOV8 = 3,
	TENSORRT = 4,
	OPENVINO = 5,
	FACEDETECT = 6,
	FACERECOGNIZE = 7,
	ALPR = 8,
	OCR = 9,
	ANOMALIB = 10,
	POSE = 11,
	SAM = 12,
	ODHUBMODEL = 13,
	YOLOV10RTOD = 14, // TensorRT Object Detection for Yolov10
	YOLOV10OVOD = 15, // OpenVINO Object Detection for Yolov10
	CUSTOMDETECTOR = 16, // Custom Detector
	YOLOV12 = 17 // YoloV12 standard for yolov12
};
namespace ANSCENTER {
    struct Region {
        int regionType; // 0: rectangle, 1: polygon, 2: line
        std::string regionName;
        std::vector<cv::Point> polygon;
    };
    struct ParamType {
        int type; // Type (0: int, 1: double, etc.)
        std::string name;
        std::string value;
    };
    struct Params {
        std::string handleName;
        int handleId;
        std::vector<ParamType> handleParametersJson; // Now a list of ParamType objects
        std::vector<Region> ROIs;         // Supports named/complex ROIs
    }; 
    /* Example
    {
          "parameters": [
            {
              "handleName": "VehicleDetector",
              "handleId": 101,
              "handleParametersJson": [
                {
                  "type": 0, // int
                  "name": "modelType",
                  "value": 1
                }
              ],
              "ROIs": [
                {
                  "regionType": 1,
                  "regionName": "Area1",
                  "polygon": [
                    {"x": 100, "y": 100},
                    {"x": 200, "y": 100},
                    {"x": 200, "y": 200},
                    {"x": 100, "y": 200}
                  ]
                }
              ]
            },
            {
              "handleName": "PersonDetector",
              "handleId": 102,
              "handleParametersJson": [
                {
                  "type": 1, // double
                  "name": "thesHold",
                  "value": 0.5
                }
              ],
              "ROIs": []
            }
          ]
        }
    */
	struct Object
	{
		int classId{ 0 };
		int trackId{ 0 };
		std::string className{};
		float confidence{ 0.0 };
		cv::Rect box{};
		std::vector<cv::Point2f> polygon;    // polygon that contain x1 ,y1,x2,y2,x3,y3,x4,y4
		cv::Mat mask{};                      // image in box (cropped) 
		std::vector<float> kps{};            // Pose exsimate keypoint or bouding box
		std::string extraInfo;               // More information such as facial recognition
		std::string cameraId;                // Use to check if this object belongs to any camera
	};
	class ANSLIB_API ANSLIB {
    public:
        ANSLIB();
        ~ANSLIB();
		int Initialize(const char* licenseKey,
			const char* modelFilePath,
			const char* modelFileZipPassword,
			float modelThreshold,
			float modelConfThreshold,
			float modelNMSThreshold,
			int modelType,
			int detectionType, std::string &labels);
		int RunInference(cv::Mat cvImage, const char* cameraId, std::vector<ANSCENTER::Object>& detectionResult);
		int OptimizeModel(const char* modelFilePath, const char* modelFileZipPassword, int modelType, int fp16);
		int Optimize(bool fp16); // Perform optimization on the loaded model on current model folder
		int GetEngineType();
		int LoadModelFromFolder(const char* licenseKey, const char* modelName, const char* className,
			float detectionScoreThreshold, float modelConfThreshold, float modelMNSThreshold,
			int autoDetectEngine, int modelType, int detectionType, const char* modelFolder, std::string& labelMap);
		int DetectMovement(cv::Mat image, const char* cameraId, std::vector<Object>& results);
		cv::Rect GetActiveWindow(cv::Mat cvImage);
    private:
        HMODULE dllHandle = nullptr;
        bool loaded = false;
        void* ANSHandle = nullptr;
        const char* CreateANSODHandle_CS(void** Handle,
            const char* licenseKey,
            const char* modelFilePath,
            const char* modelFileZipPassword,
            float modelThreshold,
            float modelConfThreshold,
            float modelNMSThreshold,
            int autoDetectEngine,
            int modelType,
            int detectionType);
        int RunInferenceComplete_CPP(void** Handle, cv::Mat** cvImage, const char* cameraId, std::vector<ANSCENTER::Object>& detectionResult);
        const char* OptimizeModelStr_CS(const char* modelFilePath, const char* modelFileZipPassword, int modelType, int fp16);
        int ReleaseANSODHandle(void** Handle);
		int GetActiveRect(cv::Mat cvImage, cv::Rect& activeWindow);
		int GetODParameters(std::vector<ANSCENTER::Params>& param);
        bool IsLoaded() const;

        typedef const char* (*CreateANSODHandle_CSFuncT)(void**, const char*, const char*, const char*, float, float, float, int, int, int);
        typedef const char* (*OptimizeModelStr_CSFuncT)(const char*, const char*, int, int);
        typedef int (*RunInferenceComplete_CPPFuncT)(void**, cv::Mat**, const char*, std::vector<ANSCENTER::Object>& detectionResult);
        typedef int (*ReleaseANSODHandleFuncT)(void**);
		typedef int(*GetEngineTypeFuncT)();
		typedef int(*LoadModelFromFolderFuncT)(void**, const char*, const char*, const char*, float, float, float, int, int, int, const char*,std::string&);
		typedef int(*GetActiveRectFuncT)(void**, cv::Mat, cv::Rect&);
		typedef int(*DetectMovementFuncT)(void**, cv::Mat, const char*, std::vector<ANSCENTER::Object>&);
		typedef int(*OptimizeFuncT)(void**, bool);
        typedef int(*GetODParametersFuncT)(void**, std::vector<ANSCENTER::Params>&);

        CreateANSODHandle_CSFuncT CreateANSODHandle_CSFunc = nullptr;
        OptimizeModelStr_CSFuncT OptimizeModelStr_CSFunc = nullptr;
        ReleaseANSODHandleFuncT ReleaseANSODHandleFunc = nullptr;
        RunInferenceComplete_CPPFuncT RunInferenceComplete_CPPFunc = nullptr;
		GetEngineTypeFuncT GetEngineTypeFunc = nullptr;
		LoadModelFromFolderFuncT LoadModelFromFolderFunc = nullptr;
		GetActiveRectFuncT GetActiveRectFunc = nullptr;
		DetectMovementFuncT DetectMovementFunc = nullptr;
		OptimizeFuncT OptimizeFunc = nullptr;
        GetODParametersFuncT GetODParametersFunc = nullptr;
	};
}
#endif 
