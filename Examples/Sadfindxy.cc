#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include"Sadfind.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    Mat leftimg=imread("/home/jinln/jinln/DATASET/baslerstereodata/20201019/2020101900/enhance/image_0/000000.png");
    Mat rightimg=imread("/home/jinln/jinln/DATASET/baslerstereodata/20201019/2020101900/enhance/image_2/000000.png");

    Sadfind(leftimg,rightimg);

    return 0;
}
