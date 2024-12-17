#include <iostream>
#include <array>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "inference.hpp"
#include "detection.hpp"

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

using namespace std;
namespace fs = std::filesystem;

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main(int argc, char *argv[])
{
    string path = argv[1];
    vector<cv::Mat> image_list;

    try
    {
        if (fs::exists(path) && fs::is_directory(path))
        {
            for (const auto &entry : fs::directory_iterator(path))
            {
                if (entry.is_regular_file())
                {
                    const string filepath = entry.path().c_str();
                    cv::Mat image = cv::imread(filepath);
                    image_list.push_back(image);
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
    }

    

    return 0;
}