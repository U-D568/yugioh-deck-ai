#include <cmath>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>

#include "cnpy.h"
#include "utils.hpp"

#define VECTOR_SIZE 1000

using namespace std;

double euclidean_distance(float* vector1, float* vector2, int length)
{
    float distance = 0;
    for (int i = 0; i < length; i++) {
        distance += pow(vector1[i] - vector2[i], 2);
    }
    return (float)sqrt(distance);
}

static bool sort_by_distance(const tuple<int, float>& e1, const tuple<int, float>& e2)
{
    float val1 = get<1>(e1);
    float val2 = get<1>(e2);
    return (val1 < val2);
}

Matrix::Matrix(char* matrix_path, char* id_list_path)
{
    // read *.npy file
    matrix = cnpy::npy_load(matrix_path);
    rows = matrix.shape[0];
    cols = matrix.shape[1];

    // read and make id_list
    ifstream ifs;
    ifs.open(id_list_path);
    if (ifs.is_open())
    {
        string line;
        while (getline(ifs, line))
        {
            int id = stoi(line);
            id_list.push_back(id);
        }
        ifs.close();
    }
}

vector<int> Matrix::get_card_id(vector<float> pred)
{
    float* matrix_data = matrix.data<float>();
    vector<tuple<int, float>> distance_list; // <index, distance>

    for (int i = 0; i < rows; i++)
    {
        int index = i * cols;
        float* embedding = matrix_data + index;
        float distance = euclidean_distance(&pred[0], embedding, cols);

        distance_list.push_back(make_tuple(i, distance));
    }

    sort(distance_list.begin(), distance_list.end(), sort_by_distance);

    vector<int> result = {
        id_list[get<0>(distance_list[0])],
        id_list[get<0>(distance_list[1])],
        id_list[get<0>(distance_list[2])]
    };
    return result;
}

float BBox::get_area()
{
    return (y2 - y1) * (x2 - x1);
}

float iou(BBox box1, BBox box2)
{
    const float eps = 1e-6;
    float iou = 0.f;

    float x1 = std::max(box1.x1, box2.x1);
    float x2 = std::min(box1.x2, box2.x2);
    float y1 = std::max(box1.y1, box2.y1);
    float y2 = std::min(box1.y2, box2.y2);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    iou = inter / (box1.get_area() + box2.get_area() - inter + eps);

    return iou;
}

vector<BBox> nms(vector<BBox> boxes, float iou_thresh)
{
    vector<BBox> result;
    sort(boxes.begin(), boxes.end(), [](BBox box1, BBox box2) {
        return box1.score > box2.score;
    });

    for (int i = 0; i < boxes.size(); i++)
    {
        if (boxes[i].score == -1)
            continue;
        for (int j = i + 1; j < boxes.size(); j++)
        {
            if (iou(boxes[i], boxes[j]) > iou_thresh)
                boxes[j].score = -1;
        }
    }

    for (BBox bbox : boxes)
    {
        if (bbox.score == -1)
            continue;
        result.push_back(bbox);
    }

    // boxes.erase(remove_if(boxes.begin(), boxes.end(), [](BBox val){
    //     return val.score != -1;
    // }), boxes.end());

    return result;
}

Padding square_padding(int width, int height, int size)
{
    Padding padding;
    float dw = (size - width) / 2.0;
    float dh = (size - height) / 2.0;

    padding.top = (int)round(dh - 0.1);
    padding.bottom = (int)round(dh + 0.1);
    padding.left = (int)round(dw - 0.1);
    padding.right = (int)round(dw + 0.1);

    return padding;
}
