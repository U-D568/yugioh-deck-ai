#include <iostream>
#include <vector>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <tuple>
#include <string>
#include <cstdio>
#include <sstream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "inference.hpp"
#include "detection.hpp"
#include "utils.hpp"
#include "nlohmann/json.hpp"

#define BUFFER_SIZE 1024

using namespace std;

void handle_client_request(int client_socket, Model &model, Detector &detector, Matrix matrix)
{
    uint32_t image_size;
    recv(client_socket, &image_size, sizeof(image_size), 0);
    image_size = ntohl(image_size);

    if (image_size == 0)
    {
        close(client_socket);
        return;
    }

    vector<uint8_t> buffer;
    buffer.resize(image_size);
    ssize_t total_received = 0;
    ssize_t bytes_received = 0;

    while (total_received < image_size)
    {
        ssize_t data_size = min((long)BUFFER_SIZE, image_size - total_received);
        bytes_received = recv(client_socket, buffer.data() + total_received, BUFFER_SIZE, 0);
        if (bytes_received < 0)
        {
            cerr << "Error has occured while receiving data. (client socket: " << client_socket << ")" << endl;
            close(client_socket);
            return;
        }
        else if (bytes_received == 0)
            break;
        total_received += bytes_received;
    }

    // Make image from bytes
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    // Object detection
    clock_t start = clock();
    auto bboxes = detector.inference(image);
    clock_t end = clock();

    cout << "object detection duration: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    // cropping bounding boxes
    vector<cv::Mat> image_list;
    for (int i = 0; i < bboxes.size(); i++)
    {
        BBox bbox = bboxes[i];

        int width = bbox.x2 - bbox.x1;
        int height = bbox.y2 - bbox.y1;

        cv::Rect rect(bbox.x1, bbox.y1, width, height);
        cv::Mat cropped_ref(image, rect);
        cv::Mat cropped;
        cropped_ref.copyTo(cropped);

        image_list.push_back(cropped);
    }

    // Predcit card image
    vector<EmbeddingTensor> prediction = model.inference(image_list, 16);

    ostringstream oss; // Response string stream
    for (int i = 0; i < prediction.size(); i++)
    {
        EmbeddingTensor pred = prediction[i];
        vector<int> id_list = matrix.get_card_id(pred.value);
        
        nlohmann::json js;
        js["id_list"] = {id_list[0], id_list[1], id_list[2]};
        js["xyxy"] = {bboxes[i].x1, bboxes[i].y1, bboxes[i].x2, bboxes[i].y2};
        oss << js.dump();
    }

    string message = oss.str();
    send(client_socket, message.c_str(), message.size(), 0);
    close(client_socket);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        exit(1);
    }

    char *model_path = argv[1];
    char *detector_path = argv[2];
    char *matrix_path = argv[3];
    char *id_list_path = argv[4];
    char *test_image = argv[5];

    Model model = Model(model_path);
    Matrix matrix = Matrix(matrix_path, id_list_path);
    Detector detector = Detector(detector_path);

    // Create Socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);

    // Sepcifiying the address
    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(9000);
    server_address.sin_addr.s_addr = INADDR_ANY;

    // Binding socket
    bind(server_socket, (struct sockaddr *)&server_address, sizeof(server_address));

    // Listening to the assigned socket
    listen(server_socket, 10);

    // Make new thread to handle client request
    try
    {
        while (true)
        {
            int client_socket = accept(server_socket, nullptr, nullptr);

            thread th = thread(handle_client_request, client_socket, ref(model), ref(detector), matrix);
            th.detach();
        }
    }
    catch (const exception &err)
    {
        cerr << err.what() << endl;
    }

    close(server_socket);

    return 0;
}