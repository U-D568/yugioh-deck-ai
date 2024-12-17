#include <cstdio>
#include <array>
#include <limits>
#include <tuple>
#include <ctime>

#include <opencv2/opencv.hpp>
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include "cnpy.h"
#include "inference.hpp"
#include "utils.hpp"

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

using namespace std;

Model::Model(char *model_path)
{
    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_path);
    TFLITE_MINIMAL_CHECK(model != nullptr);
}

vector<EmbeddingTensor> Model::inference(vector<cv::Mat> input_data, int batch_size)
{
    // Initiate Interpreter
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // interpreter->SetNumThreads(4);

    // Get Input Tensor Dimensions
    int input_index = interpreter->inputs()[0];
    int height = interpreter->tensor(input_index)->dims->data[1];
    int width = interpreter->tensor(input_index)->dims->data[2];
    int channel = interpreter->tensor(input_index)->dims->data[3];

    // initialize variables
    vector<float> flatten;
    vector<EmbeddingTensor> result;
    int prev_chunk_size = -1;

    // Preprocessing
    for (int i = 0; i < input_data.size(); i++)
    {
        cv::Mat image = input_data[i];

        cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1 / 255.f, 0);

        if (image.isContinuous())
            flatten.insert(flatten.end(), image.ptr<float>(0), image.ptr<float>(image.rows - 1) + image.cols * image.channels());
        else
        {
            cout << "isContinuous" << endl;
            for (int i = 0; i < image.rows; i++)
                flatten.insert(flatten.end(), image.ptr<float>(i), image.ptr<float>(i) + image.cols * image.channels());
        }
    }

    int image_size = height * width * channel;
    for (int i = 0; i < input_data.size(); i += batch_size)
    {
        int chunk_size = std::min(batch_size, (int)input_data.size() - i);
        // Copy image to input tensor
        if (prev_chunk_size != chunk_size)
        {
            // resize input tensor and re-allocate tensors
            prev_chunk_size = chunk_size;
            vector<int> new_shape = {chunk_size, height, width, channel};
            interpreter->ResizeInputTensorStrict(input_index, new_shape);
            TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
        }

        float *input_tensor = interpreter->typed_input_tensor<float>(0);
        memcpy(input_tensor, &flatten[i], image_size * chunk_size * sizeof(float));

        // Inference
        clock_t start = clock();
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        clock_t end = clock();

        cout << "embedding duration: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

        // Get Output
        float *output_tensor = interpreter->typed_output_tensor<float>(0);
        int output_index = interpreter->outputs()[0];
        auto length = interpreter->tensor(output_index)->dims->data[1];

        for (int j = 0; j < chunk_size; j++)
        {
            EmbeddingTensor embedding;
            auto start_address = output_tensor + length * j;
            embedding.value.assign(start_address, start_address + length);
            result.push_back(embedding);
        }
    }

    return result;
}