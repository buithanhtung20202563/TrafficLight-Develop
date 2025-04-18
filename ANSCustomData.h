#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "ANSLIB.h"

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