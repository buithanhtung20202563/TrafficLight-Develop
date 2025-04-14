#ifndef ANSCUSTOMTL_H
#define ANSCUSTOMTL_H
#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "ANSLIB.h"
#define CUSTOM_API __declspec(dllexport)
struct CustomObject
{
    int classId{ 0 };
    int trackId{ 0 };
    std::string className{};
    float confidence{ 0.0 };
    cv::Rect box{};
    std::vector<cv::Point2f> polygon;
    cv::Mat mask;             //Json string mask ="point1.x,point1.y,...."
    std::vector<float> kps{};   // Pose exsimate keypoint
    std::string extraInfo;      // More information such as facial recognition
    std::string cameraId;
};
struct CustomRegion {
    int regionType; // 0: rectangle, 1: polygon, 2: line
    std::string regionName;
    std::vector<cv::Point> polygon;
};
struct CustomParamType {
    int type; // Type (0: int, 1: double, etc.)
    std::string name;
    std::string value;
};
struct CustomParams {
    std::string handleName;
    int handleId;
    std::vector<CustomParamType> handleParametersJson; // Now a list of ParamType objects
    std::vector<CustomRegion> ROIs;         // Supports named/complex ROIs
};

/* Example
{
      "parameters": [
        {
          "handleName": "VehicleDetector",
          "handleId": 0,
          "handleParametersJson": [
            {
              "type": 0, // int
              "name": "modelType",
              "value": 1
            },
            {
             "type": 1, // double
              "name": "threshold",
              "value": 0.7
            },
            {
              "type": 4, // list
              "name": "objects",
              "value": "car,truck,bus"
            }

          ],
          "ROIs": [
            {
			  "regionType": 1,// Rectangle
              "regionName": "DetectArea",
              "polygon": [
                {"x": 100, "y": 100},
                {"x": 200, "y": 100},
                {"x": 200, "y": 200},
                {"x": 100, "y": 200}
              ]
            },
            {
              "regionType": 0,// Line
              "regionName": "CrossingLine",
              "polygon": [
                {"x": 100, "y": 100},
                {"x": 200, "y": 100}
              ]
			},
			{
			  "regionType": 4,// Right Line direction
			  "regionName": "Direction",
			  "polygon": [
				{"x": 100, "y": 100},
				{"x": 200, "y": 100}
			  ]
			}
          ]
        },
        {
          "handleName": "TrafficLight",
          "handleId": 1,
          "handleParametersJson": [
            {
              "type": 1, // double
              "name": "thesHold",
              "value": 0.5
            }
          ],
          "ROIs": [
               {
			  "regionType": 1,// Rectangle
              "regionName": "TrafficRoi",
              "polygon": [
                {"x": 100, "y": 100},
                {"x": 200, "y": 100},
                {"x": 200, "y": 200},
                {"x": 100, "y": 200}
              ]
            }
          ]
        }
      ]
    }
*/

// Interface
class CUSTOM_API IANSCustomClass
{
protected:
  std::string _modelDirectory; // The directory where the model is located
  float _detectionScoreThreshold{0.5};
  std::vector<CustomParams> _param; // Parameters for the model
public:
  virtual bool Initialize(const std::string &modelDirectory, float detectionScoreThreshold, std::string &labelMap) = 0;
  virtual bool OptimizeModel(bool fp16) = 0;
  virtual std::vector<CustomObject> RunInference(const cv::Mat &input) = 0;
  virtual std::vector<CustomObject> RunInference(const cv::Mat &input, const std::string &camera_id) = 0;
  virtual bool ConfigureParamaters(std::vector<CustomParams> &param) = 0;
  bool SetParamaters(const std::vector<CustomParams> &param)
  {
    try
    {
      _param.clear();
      if (param.empty())
        return false;
      for (const auto &p : param)
      {
        _param.push_back(p);
      }
      return true;
    }
    catch (...)
    {
      return false;
    }
  };
  virtual bool Destroy() = 0;
};

// Implementation of the traffic light model
class CUSTOM_API ANSCustomTL : public IANSCustomClass
{

private:
  std::recursive_mutex _mutex;
  ANSCENTER::ANSLIB vehicleDetector;      // This is the vehicle object detector
  ANSCENTER::ANSLIB trafficLightDetector; // This is the traffic light object detector

  // Store label maps for vehicle and traffic light
  std::string _vehicleLabelMap;
  std::string _trafficLightLabelMap;

  std::string _vehicleClassName;
  std::string _trafficLightClassName;

  int _vehicleModelType;
  int _vehicleDetectionType;

  int _trafficLightModelType;
  int _trafficLightDetectionType;

  std::string _vehicleModelName;
  std::string _trafficLightModelName;

  double _detectionScoreThreshold{ 0.5 };
public:
    bool Initialize(const std::string& modelDiretory, float detectionScoreThreshold, std::string& labelMap)override;
    bool OptimizeModel(bool fp16)override;
    //bool SetParameter(std::string parameters);
    std::vector<CustomObject> RunInference(const cv::Mat& input)override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& camera_id)override;
    bool ConfigureParamaters(std::vector<CustomParams>& param)override;

    bool Destroy()override;
    ANSCustomTL();
    ~ANSCustomTL();
};
#endif