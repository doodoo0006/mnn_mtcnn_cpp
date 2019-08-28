#include "imgProcess.h"
#include "mtcnn.h"

int detFace(cv::Mat input_rgb, const char* path, std::vector<Bbox> &finalBbox)
{
    static MTCNN mtcnn;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        mtcnn.load(model_path);
        is_model_prepared = true;
    }

    mtcnn.detect(input_rgb, finalBbox, 0.25f);

    return 0;
}

int deal(cv::Mat& frame, const char* path)
{
    cv::flip(frame, frame, 1);

    cv::Mat input_rgb;
    cv::cvtColor(frame, input_rgb, cv::COLOR_BGRA2RGB);
	input_rgb = input_rgb.t();

	std::vector<Bbox> finalBbox;
	detFace(input_rgb, path, finalBbox);

	for (int i = 0; i < finalBbox.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 0,   0, 255);

        for (int s = 0; s < 5; s++)
        {
            cv::Point2f pt(finalBbox[i].ppoint[s + 5], finalBbox[i].ppoint[s]);
            cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0, 255), cv::FILLED);
        }

		cv::Rect rs(finalBbox[i].rect().y,      finalBbox[i].rect().x,
		            finalBbox[i].rect().height, finalBbox[i].rect().width);

		cv::rectangle(frame, rs, color, 1);
	}

    return 0;
}