#pragma once

#include <iostream>
#include <vector>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include <opencv2/opencv.hpp>

#include "utils.hpp"

using namespace std;

class Detector
{
private:
    unique_ptr<tflite::FlatBufferModel> model;
    cv::Mat preprocessing(cv::Mat image);
    vector<BBox> postprocessing(float* output_tensor, int rows, int image_width, int image_height);

public:
    Detector(char* model_path);
    vector<BBox> inference(cv::Mat image);
};
