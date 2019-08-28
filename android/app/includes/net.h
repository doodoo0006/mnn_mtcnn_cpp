#ifndef _NET_H_
#define _NET_H_

#include <vector>
#include <string>
#include <ImageProcess.hpp>
#include <Interpreter.hpp>
#include <Tensor.hpp>
#include <memory>

#include "opencv2/opencv.hpp"

class Inference_engine_tensor
{
public:
    Inference_engine_tensor()
    { }

    ~Inference_engine_tensor()
    { }

    void add_name(std::string &layer)
    {
        layer_name.push_back(layer);
    }

    float* score(int idx)
    {
        return (float*)out_feat[idx].data;
    }

public:
    std::vector<std::string> layer_name;
    std::vector<cv::Mat> out_feat;
};

class Inference_engine
{
public:
    Inference_engine();
    ~Inference_engine();

    int load_param(std::vector<std::string> &file, int num_thread = 1);
    int set_params(int inType, int outType, std::vector<float> &mean, std::vector<float> &scale);
    int infer_img(cv::Mat& imgs, Inference_engine_tensor& out);
    int infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Inference_engine_tensor>& out);
private: 
    MNN::Interpreter* netPtr;
	MNN::Session* sessionPtr;
    MNN::CV::ImageProcess::Config config;
};
#endif