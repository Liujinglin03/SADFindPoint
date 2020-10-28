

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include<vector>

using namespace cv;

struct SAVESAD{
    float sad;
    int x;
    int y;
};

bool cmpare( const SAVESAD sad1,const SAVESAD sad2)
{
    return sad1.sad<sad2.sad;
}

void Sadfind(Mat& left,Mat& right);

// NCC
void compute_sq(IplImage* left_img, IplImage* right_img, float *left_sq, float *right_sq, float *left_avg, float *right_avg);
