#ifndef ANSCUSTOMTL_H
#define ANSCUSTOMTL_H
#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "ANSLIB.h"
#include "ANSCustomData.h"
#include "Vehicle.h"
#include "TrafficLight.h"

#define CUSTOM_API __declspec(dllexport)

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
  // virtual bool SetParamaters(const std::vector<CustomParams>& param) = 0;
  //{
  // try {
  //     _param.clear();
  //     if (param.empty())return false;
  //     for (const auto& p : param) {
  //         _param.push_back(p);
  //     }
  //     return true;
  // }
  // catch (...) {
  //     return false;
  // }
  //};
  virtual bool Destroy() = 0;

};

// Implementation of the traffic light model
class CUSTOM_API ANSCustomTL : public IANSCustomClass
{

private:
  CACVehicle m_cVehicleDetector;           // Vehicle object
  CACTrafficLight m_cTrafficLightDetector; // Traffic light object
  std::recursive_mutex _mutex;
  ANSCENTER::ANSLIB vehicleDetector; // This is the vehicle object detector
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
    bool SetParamaters(const std::vector<CustomParams>& param);
    std::vector<CustomObject> RunInference(const cv::Mat& input)override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& camera_id)override;
    bool ConfigureParamaters(std::vector<CustomParams>& param)override;

    bool Destroy()override;
    ANSCustomTL();
    ~ANSCustomTL();
};
#endif