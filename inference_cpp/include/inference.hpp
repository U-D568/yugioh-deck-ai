#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "utils.hpp"


using namespace std;

class Model
{
private:
    unique_ptr<tflite::FlatBufferModel> model;

public:
    Model(char* model_path);
    vector<EmbeddingTensor> inference(vector<cv::Mat> image, int batch_size);
};
