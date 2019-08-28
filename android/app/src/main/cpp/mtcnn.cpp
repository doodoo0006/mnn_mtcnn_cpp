#include <algorithm>
#include <map>
#include "mtcnn.h"

bool cmpScore(Bbox lsh, Bbox rsh)
{
    bool res = lsh.score < rsh.score;
    return res;
}

bool cmpArea(Bbox lsh, Bbox rsh)
{
    bool res = lsh.area() >= rsh.area();
    return res;
}

MTCNN::MTCNN()
{ }

MTCNN::~MTCNN()
{ }

void MTCNN::load(const std::string &model_path, int num_thread)
{
    std::vector<float> mean_vals{ 127.5, 127.5, 127.5 };
    std::vector<float> norm_vals{ 0.0078125, 0.0078125, 0.0078125 };

    std::vector<std::string> tmpp = { model_path + "/det1.mnn" };
    Pnet.load_param(tmpp, num_thread);
    Pnet.set_params(1, 1, mean_vals, norm_vals);

    std::vector<std::string> tmpr = { model_path + "/det2.mnn" };
    Rnet.load_param(tmpr, num_thread);
    Rnet.set_params(1, 1, mean_vals, norm_vals);

    std::vector<std::string> tmpo = { model_path + "/det3.mnn" };
    Onet.load_param(tmpo, num_thread);
    Onet.set_params(1, 1, mean_vals, norm_vals);
}

void MTCNN::generateBbox(cv::Mat score, cv::Mat location, std::vector<Bbox>& boundingBox_, float scale)
{
    const int stride = 2;
    const int cellsize = 12;

    int sc_rows, sc_cols;
    if ( 4 == score.dims)
    {
        sc_rows = score.size[2];
        sc_cols = score.size[3];
    }

    float* p  = (float *)score.data + sc_rows * sc_cols;
    float inv_scale = 1.0f / scale;
    for(int row = 0; row < sc_rows; row++)
    {
        for(int col = 0; col < sc_cols; col++)
        {
            Bbox bbox;
            if( *p > threshold[0] )
            {
                bbox.score = *p;
                bbox.x1 = round((stride * col + 1) * inv_scale);
                bbox.y1 = round((stride * row + 1) * inv_scale);
                bbox.x2 = round((stride * col + 1 + cellsize) * inv_scale);
                bbox.y2 = round((stride * row + 1 + cellsize) * inv_scale);
                const int index = row * sc_cols + col;
                for(int channel = 0;channel < 4; channel++)
                {
                    float* tmp = (float *)(location.data) + channel * sc_rows * sc_cols;
                    bbox.regreCoord[channel] = tmp[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
        }
    }

    return;
}

void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname)
{
    if (boundingBox_.empty()) return;

    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float iou = 0; float maxX = 0; float maxY = 0; float minX = 0; float minY = 0;

    std::vector<int> vPick; int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i)
    {
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
    }

    if (1 == vScores.size())
    {
        vPick[0] = vScores.rbegin()->second;
        nPick += 1;
    }

    while(vScores.size() > 1)
    {
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;

        if (nPick >= vPick.size())break;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();)
        {
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);

            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;

            iou = maxX * maxY;
            if (!modelname.compare("Union"))
            {
                iou = iou / (boundingBox_.at(it_idx).area() + boundingBox_.at(last).area() - iou);
            }
            else if(!modelname.compare("Min"))
            {
                iou = iou / ((boundingBox_.at(it_idx).area() < boundingBox_.at(last).area())
                    ? boundingBox_.at(it_idx).area() : boundingBox_.at(last).area());
            }

            if(iou > overlap_threshold)
            {
                it = vScores.erase(it);
                if ( vScores.size() <= 1)
                {
                    break;
                }
            }
            else
            {
                it++;
            }
        }
    }
    
    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++)
    {
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}

void MTCNN::refine(std::vector<Bbox>& vecBbox, const int& height, const int& width, bool square)
{
    if (vecBbox.empty())return;
    
    float bbw = 0, bbh = 0, max_side = 0;
    float h = 0, w = 0;
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    for (auto it = vecBbox.begin(); it != vecBbox.end(); it++)
    {
        bbw = it->x2 - it->x1 + 1;
        bbh = it->y2 - it->y1 + 1;

        x1 = it->x1 + bbw * it->regreCoord[1];
        y1 = it->y1 + bbh * it->regreCoord[0];
        x2 = it->x2 + bbw * it->regreCoord[3];
        y2 = it->y2 + bbh * it->regreCoord[2];

        if(square)
        {
              w = x2 - x1 + 1;
              h = y2 - y1 + 1;
              int maxSide = ( h > w ) ? h:w;
              x1 = x1 + w * 0.5 - maxSide * 0.5;
              y1 = y1 + h * 0.5 - maxSide * 0.5;
              x2 = round(x1 + maxSide - 1);
              y2 = round(y1 + maxSide - 1);
              x1 = round(x1);
              y1 = round(y1);
         }

        it->x1 = x1 < 0 ? 0 : x1;
        it->y1 = y1 < 0 ? 0 : y1;
        it->x2 = x2 >= width  ? width  - 1 : x2;
        it->y2 = y2 >= height ? height - 1 : y2;
    }
}

void MTCNN::PNet()
{
    int img_h = img.rows;
    int img_w = img.cols;

    if (scales_.empty())
    {
        const int min_det_size = 12;
        const int minsize      = 40;
        const float pre_facetor = 0.5f;
        float minl = img_w < img_h ? img_w: img_h;
        float m = (float)min_det_size / minsize;
        minl *= m;
        float factor = pre_facetor;
        while(minl > min_det_size)
        {
            scales_.push_back(m);
            minl *= factor;
            m = m * factor;
        }
    }

    firstBbox_.clear();
    for (size_t i = 0; i < scales_.size(); i++)
    {
        int hs = (int)ceil(img_h * scales_[i]);
        int ws = (int)ceil(img_w * scales_[i]);

        cv::Mat in;
        cv::resize(img, in, cv::Size(ws, hs));

        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv4-2";
        out.add_name(tmp_str1);

        Pnet.infer_img(in, out);

        std::vector<Bbox> boundingBox_;
        generateBbox(out.out_feat[0], out.out_feat[1], boundingBox_, scales_[i]);
        nms(boundingBox_, 0.7);
        firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
    return;
}
void MTCNN::RNet()
{
    secondBbox_.clear();
    int count = 0;
    for(std::vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++)
    {
        cv::Rect rs = it->rect();
        cv::Mat in = img(rs).clone();

        cv::resize(in, in, cv::Size(24, 24));

        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv5-2";
        out.add_name(tmp_str1);
        Rnet.infer_img(in, out);

        float* thresh = (float*)out.out_feat[0].data;
        float* bbox   = (float*)out.out_feat[1].data;
        Bbox tmp_box = *it;
        if ( thresh[1] > threshold[1] )
        {
            for (int channel = 0; channel < 4; channel++)
            {
                tmp_box.regreCoord[channel] = bbox[channel];
            }
            tmp_box.score = thresh[1]; 
            secondBbox_.push_back(tmp_box);
        }
    }
}

void MTCNN::ONet()
{
    thirdBbox_.clear();
    for(std::vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++)
    {
        cv::Rect rs = it->rect();
        cv::Mat in = img(rs).clone();
        cv::resize(in, in, cv::Size(48, 48));

        Inference_engine_tensor out;
        std::string tmp_str = "prob1";
        out.add_name(tmp_str);

        std::string tmp_str1 = "conv6-2";
        out.add_name(tmp_str1);

        std::string tmp_str2 = "conv6-3";
        out.add_name(tmp_str2);

        Onet.infer_img(in, out);

        float* thresh = (float*)out.out_feat[0].data;
        float* bbox   = (float*)out.out_feat[1].data;
        float* keyPoint = (float*)out.out_feat[2].data;

        if (thresh[1] > threshold[2])
        {
            for (int channel = 0; channel < 4; channel++)
            {
                it->regreCoord[channel] = bbox[channel];
            }
            it->score = thresh[1];
            for (int num = 0; num < 5; num++)
            {
                (it->ppoint)[num]     = it->x1 + it->width()  * keyPoint[num + 5] - 1;
                (it->ppoint)[num + 5] = it->y1 + it->height() * keyPoint[num] - 1;
            }
            thirdBbox_.push_back(*it);
        }
    }
}

void MTCNN::detectInternal(cv::Mat& img_, std::vector<Bbox>& finalBbox_)
{
    const float nms_threshold[3] = {0.7f, 0.7f, 0.7f};

    img = img_;
    PNet();
    if ( !firstBbox_.empty())
    {
        nms(firstBbox_, nms_threshold[0]);
        refine(firstBbox_, img_.rows, img_.cols, true);

        RNet();
        if( !secondBbox_.empty())
        {
            nms(secondBbox_, nms_threshold[1]);
            refine(secondBbox_, img_.rows, img_.cols, true);

            ONet();
            if ( !thirdBbox_.empty())
            {
                refine(thirdBbox_, img_.rows, img_.cols, false);

                std::string ts = "Min";
                nms(thirdBbox_, nms_threshold[2], ts);
            }
        }
    }

    finalBbox_ = thirdBbox_;
}

void MTCNN::detect(const cv::Mat& img_, 
                   std::vector<Bbox>& finalBbox, 
                   float scale,
                   cv::Rect detRegion)
{   
    if (img_.empty()) return;

    cv::Mat testImg;
    if (detRegion.x > 0  && detRegion.y > 0
        && detRegion.br().x < img_.cols && detRegion.br().y < img_.rows)
    {
        testImg = img_(detRegion).t();
    }
    else
    {
        testImg = img_.t();
    }

    if ( scale < 1.0f && scale > 0.0f )
    {
        cv::resize(testImg, testImg, cv::Size(), scale, scale);
    }
    
    detectInternal(testImg, finalBbox);

    for (int i = 0; i < finalBbox.size(); i++)
    {
        if ( scale < 1.0f && scale > 0.0f )
        {
            finalBbox[i].scale_inverse(scale);
        }

        finalBbox[i] = finalBbox[i].trans();

        for (int ii = 0; ii < 5; ii++)
        {
            finalBbox[i].ppoint[ii]     += detRegion.x;
            finalBbox[i].ppoint[ii + 5] += detRegion.y;
        }

        finalBbox[i].x1 += detRegion.x;
        finalBbox[i].y1 += detRegion.y;
        finalBbox[i].x2 += detRegion.x;
        finalBbox[i].y2 += detRegion.y;
    }

    return;
}

