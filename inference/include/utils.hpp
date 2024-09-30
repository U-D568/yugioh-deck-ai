#pragma once

#include <vector>
#include <tuple>
#include "cnpy.h"

using namespace std;

class Matrix
{
private:
    cnpy::NpyArray matrix;
    int rows, cols;
    vector<int> id_list;

public:
    Matrix(char* matrix_path, char* id_list_path);
    vector<int> get_card_id(vector<float> pred);
};

// Bounding Box with xyxy coordinates
struct BBox
{
    int x1;
    int x2;
    int y1;
    int y2;
    float score;

    float get_area();
};

float iou(BBox box1, BBox box2);
vector<BBox> nms(vector<BBox> boxes, float iou_thresh);

struct Padding
{
    int top;
    int bottom;
    int left;
    int right;
};

Padding square_padding(int width, int height, int size);

struct EmbeddingTensor
{
    vector<float> value;
};