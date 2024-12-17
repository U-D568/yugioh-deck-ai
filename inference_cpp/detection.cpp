#include <iostream>
#include <array>
#include <vector>
#include <limits>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include "cnpy.h"
#include "detection.hpp"
#include "utils.hpp"

#define IMAGE_SIZE 640
#define GRAY 114
#define SCORE_THRESH 0.5
#define IOU_THRESH 0.7

using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

Detector::Detector(char *model_path)
{
    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_path);
    TFLITE_MINIMAL_CHECK(model != nullptr);
}

cv::Mat Detector::preprocessing(cv::Mat image)
{
    cv::Mat processed_image;
    int width = image.cols;
    int height = image.rows;

    // Image resizing
    float ratio = std::min(IMAGE_SIZE / (float)width, IMAGE_SIZE / (float)height);
    int new_width = (int)round(width * ratio);
    int new_height = (int)round(height * ratio);
    cv::resize(image, processed_image, cv::Size(new_width, new_height), cv::INTER_LINEAR);

    // Image centering and add padding
    Padding padding = square_padding(new_width, new_height, IMAGE_SIZE);

    cv::Scalar color(GRAY, GRAY, GRAY);
    cv::copyMakeBorder(processed_image, processed_image,
                       padding.top, padding.bottom, padding.left, padding.right,
                       cv::BORDER_CONSTANT, color);

    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    processed_image.convertTo(processed_image, CV_32FC3, 1.0 / 255.0, 0);
    return processed_image;
}

vector<BBox> Detector::postprocessing(float* output_tensor, int delegates, int width, int height)
{
    vector<BBox> results;
    int length = std::max(width, height);
    Padding padding = square_padding(width, height, length);

    for (int i = 0; i < delegates; i++)
    {
        float conf = output_tensor[i + delegates * 4];
        if (conf > SCORE_THRESH)
        {
            BBox xyxy;
            float x = output_tensor[i];
            float y = output_tensor[i + delegates];
            float dw = output_tensor[i + delegates * 2] / 2.0;
            float dh = output_tensor[i + delegates * 3] / 2.0;

            xyxy.x1 = std::max(round(length * (x - dw)) - padding.left, 0.f);
            xyxy.x1 = std::min(xyxy.x1, width);

            xyxy.x2 = std::min(round(length * (x + dw)) - padding.left, (float)width);
            xyxy.x2 = std::max(xyxy.x2, 0);

            xyxy.y1 = std::max(round(length * (y - dh)) - padding.top, 0.f);
            xyxy.y1 = std::min(xyxy.y1, height);

            xyxy.y2 = std::min(round(length * (y + dh)) - padding.top, (float)height);
            xyxy.y2 = std::max(xyxy.y2, 0);

            xyxy.score = conf;

            results.push_back(xyxy);
        }
    }

    results = nms(results, IOU_THRESH);

    return results;
}

vector<BBox> Detector::inference(cv::Mat src)
{
    // Initiate Interpreter
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Preprocessing
    cv::Mat image = Detector::preprocessing(src);
    int width = image.cols;
    int height = image.rows;

    // Allocate input tensor
    int input_tensor_index = interpreter->inputs()[0];
    vector<int> new_shape{1, height, width, 3};
    interpreter->ResizeInputTensor(input_tensor_index, new_shape);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    interpreter->SetAllowFp16PrecisionForFp32(true);

    // Copy image to input tensor
    float *input_tensor = interpreter->typed_input_tensor<float>(0);
    memcpy(input_tensor, image.ptr<float>(0), width * height * 3 * sizeof(float));

    // Inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Get output
    // output shape: (1, 5, 4200) (batch, xywhc, delegates)
    float *output_tensor = interpreter->typed_output_tensor<float>(0);

    int output_index = interpreter->outputs()[0];
    int delegates = interpreter->tensor(output_index)->dims->data[2];
    vector<BBox> results = Detector::postprocessing(output_tensor, delegates, src.cols, src.rows);
    return results;
}