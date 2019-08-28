#include "Bbox.h"
#include "mtcnn.h"

Bbox::Bbox()
{
    x1 = 0; y1 = 0;
    x2 = 0; y2 = 0;
    memset(ppoint,     0, 10 * sizeof(float));
    memset(regreCoord, 0, 4  * sizeof(float));
}

Bbox::~Bbox() { }

Bbox::Bbox(const Bbox& rhs)
{
    score = rhs.score;
    x1 = rhs.x1;
    x2 = rhs.x2;
    y1 = rhs.y1;
    y2 = rhs.y2;

    memcpy(this->ppoint, rhs.ppoint, 10 * sizeof(float));
    memcpy(this->regreCoord, rhs.regreCoord, 4 * sizeof(float));
}

Bbox& Bbox::operator=(const Bbox& rhs)
{
    if (this != &rhs)
    {
        this->score = rhs.score;
        this->x1 = rhs.x1;
        this->x2 = rhs.x2;
        this->y1 = rhs.y1;
        this->y2 = rhs.y2;

        memcpy(this->ppoint, rhs.ppoint, 10 * sizeof(float));
        memcpy(this->regreCoord, rhs.regreCoord, 4 * sizeof(float));
    }

    return *this;
}

cv::Rect Bbox::rect()
{
    return cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

int Bbox::width()
{
    return x2 - x1 + 1;
}

int Bbox::height()
{
    return y2 - y1 + 1;
}

float Bbox::area()
{
    return (float)(x2 - x1 + 1) * (y2 - y1 + 1);
}

cv::Point Bbox::center()
{
    return cv::Point((x1 + x2) >> 1, (y1 + y2) >> 1);
}

Bbox Bbox::trans()
{
    Bbox tmp;
    tmp.x1 = this->y1;
    tmp.x2 = this->y2;
    tmp.y1 = this->x1;
    tmp.y2 = this->x2;

    for (int i = 0; i < 5; i++)
    {
        tmp.ppoint[i] = ppoint[i + 5];
        tmp.ppoint[i + 5] = ppoint[i];
    }

    return tmp;
}

void Bbox::scale_inverse(float scale)
{
    if (scale != 0 && scale != 1)
    {
        float inv_scale = 1.0f / scale;
        x1 *= inv_scale;
        y1 *= inv_scale;
        x2 *= inv_scale;
        y2 *= inv_scale;

        for (int i = 0; i < 10; i++)
        {
            ppoint[i] *= inv_scale;
        }
    }
}
