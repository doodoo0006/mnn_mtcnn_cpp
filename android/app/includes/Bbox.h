#pragma once
#include "opencv2/opencv.hpp"

class Bbox
{
public:
    Bbox();
    ~Bbox();
    Bbox(const Bbox& rhs);
    Bbox& operator=(const Bbox& rhs);

    cv::Rect  rect();
    cv::Point center();
    int       width();
    int       height();
    float     area();

    Bbox trans();
    void scale_inverse(float scale);

public:
    float score;
    int x1, y1, x2, y2;
    float ppoint[10];
    float regreCoord[4];
};
