#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include<vector>
#include"Sadfind.h"
#include<vector>
#include<algorithm>
#include <ctime>

using namespace std;
using namespace cv;


/*
void Sadfind(Mat& preleft,Mat& preright)
{
    Mat left,right;

    preleft.copyTo(left);
    preright.copyTo(right);
    cvtColor(left,left,CV_BGR2GRAY);
    cvtColor(right,right,CV_BGR2GRAY);
    resize(right,right,Size(right.cols/2,right.rows/2),0,0,CV_INTER_AREA);

    const float w=5.0f;
    const float uR=5.0f;
    const float vR=5.0f;
    Point pr(uR,vR);
    cv::circle( right,pr,3, cv::Scalar(0,0,255),1);
    namedWindow("right img" ,0);
    imshow("right img" ,right);

    // 提取right图中，以特征点(uR,Ul)为中心, 半径为w的图像快patch
    // +1 表示最后一行和一列不算
    cv::Mat IR = right.rowRange(uR-w, uR+w+1).colRange(vR-w, vR+w+1);
    // convertTo()函数负责转换数据类型不同的Mat，即可以将类似float型的Mat转换到imwrite()函数能够接受的类型。
    IR.convertTo(IR,CV_32F);

    // 图像块均值归一化，降低亮度变化对相似度计算的影响
    IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);
    //初始化最佳相似度
    int bestDist = INT_MAX;
    int secondDist=INT_MAX;
    //初始化leftimg uL,vL
    float uL=uR,vL=vR;
    float uL1=uR,vL1=vR;
    for(int initrowpoint=w+1;initrowpoint<left.rows/2;initrowpoint++){
        for(int initcolpoint=w+1;initcolpoint<left.cols/2;initcolpoint++){
            cv::Mat IL=left.rowRange(initrowpoint-w,initrowpoint+w+1).colRange(initcolpoint-w,initcolpoint+w+1);
            IL.convertTo(IL,CV_32F);

             // 图像块均值归一化，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            // sad 计算
            float dist = cv::norm(IR,IL,cv::NORM_L1);
            if(dist<bestDist)
            {
                secondDist=bestDist;
                bestDist=dist;
                uL1=uL;vL1=vL;
                uL=initcolpoint;
                vL=initrowpoint;
            }
            else if(bestDist<dist<secondDist)
            {
                secondDist=dist;
                uL1=initcolpoint;
                vL1=initrowpoint;
            }
        }
    }
    cout<<"uL-w="<<uL-w<<"   "<<"vL-w="<<vL-w<<endl;
    Point p0(uL-w,vL-w);//bestpoint
    //Point p1(uL-w,vL-w+240);
    Point p1(uL1-w,vL1-w);
    cv::cvtColor(left, left, cv::COLOR_GRAY2BGR);
    cv::circle( left,p0,10, cv::Scalar(0,0,255),3);
    cv::circle( left,p1,10, cv::Scalar(0,0,175),2);
    imshow("left img",left);

    left.release();
    right.release();
    waitKey(0);
    return;
}
*/



void Sadfind(Mat& preleft,Mat& preright)
{
    Mat left,right;

    preleft.copyTo(left);
    preright.copyTo(right);
    cvtColor(left,left,CV_BGR2GRAY);
    cvtColor(right,right,CV_BGR2GRAY);
    resize(right,right,Size(right.cols/2,right.rows/2),0,0,CV_INTER_AREA);

    const float w=5.0f;
    const float uR=5.0f;
    const float vR=5.0f;
    Point pr(uR,vR);
    cv::circle( right,pr,3, cv::Scalar(0,0,255),1);
    namedWindow("right img" ,0);
    imshow("right img" ,right);


    // 提取right图中，以特征点(uR,Ul)为中心, 半径为w的图像快patch
    // +1 表示最后一行和一列不算
    cv::Mat IR = right.rowRange(uR-w, uR+w+1).colRange(vR-w, vR+w+1);
    // convertTo()函数负责转换数据类型不同的Mat，即可以将类似float型的Mat转换到imwrite()函数能够接受的类型。
    IR.convertTo(IR,CV_32F);

    // 图像块均值归一化，降低亮度变化对相似度计算的影响
    IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);
    //初始化最佳相似度
    int bestDist = INT_MAX;

    //初始化leftimg uL,vL
    float uL=uR,vL=vR;
    //float uL1=uR,vL1=vR;
    vector<SAVESAD> SaveSad;
    SAVESAD savepoint;

    clock_t tStart, tFinish;

    tStart = clock();
    for(int initrowpoint=w+1;initrowpoint<left.rows/2;initrowpoint++){
        for(int initcolpoint=w+1;initcolpoint<left.cols/2;initcolpoint++){
            cv::Mat IL=left.rowRange(initrowpoint-w,initrowpoint+w+1).colRange(initcolpoint-w,initcolpoint+w+1);
            IL.convertTo(IL,CV_32F);

             // 图像块均值归一化，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);


            // sad 计算
            float dist = cv::norm(IR,IL,cv::NORM_L1);

            savepoint.sad=dist;
            savepoint.x=initcolpoint;
            savepoint.y=initrowpoint;
            SaveSad.push_back(savepoint);
            if(dist<bestDist)
            {

                bestDist=dist;
                uL=initcolpoint;
                vL=initrowpoint;
            }
        }
    }

    sort(SaveSad.begin(),SaveSad.end(),cmpare);

    tFinish = clock();
    float tElapseTime = (float)(tFinish-tStart)/CLOCKS_PER_SEC;
    cout<<"消耗的时间，秒数为： "<<tElapseTime<<endl;

    cout<<"uL-w="<<uL-w<<"   "<<"vL-w="<<vL-w<<endl;
    Point p0(SaveSad[0].x-w,SaveSad[0].y-w);
    //Point p0(uL-w,vL-w);//bestpoint
    Point p1(SaveSad[1].x-w,SaveSad[1].y-w);
    Point p2(SaveSad[2].x-w,SaveSad[2].y-w);
    //Point p1(uL-w,vL-w+240);
    //Point p1(uL1-w,vL1-w);
    cv::cvtColor(left, left, cv::COLOR_GRAY2BGR);
    cv::circle( left,p0,5, cv::Scalar(0,0,255),3);
    cv::circle( left,p2,5, cv::Scalar(0,255,0),3);
    cv::circle( left,p1,5, cv::Scalar(255,0,0),3);
    imshow("left img",left);

    left.release();
    right.release();
    waitKey(0);
    return;
}




/****NCC算法，即归一化互相关匹配***********************
 *   left_sq存放了左图窗口内像素与均值差值的平方
 *   right_sq存放了右图窗口内像素与均值差值的平方
 *   left_avg存放了左图窗口内像素的均值
 *   right_avg存放了右图窗口内像素的均值
 *
 **************************************************/
void compute_sq(IplImage* left_img, IplImage* right_img, float *left_sq, float *right_sq, float *left_avg, float *right_avg)
{
    //图像的高度和宽度
    int height = left_img->height;
    int width = left_img->width;
    //窗口半径，为奇数
    int N = 5;
    //图像匹配的起始行和终止行
    int line_start = N;
    int line_end = height-N;
    //图像需要视差搜索的起始列和终止列
    int row_start = N;
    int row_end = width-N;
    int addr = 0;
    float temp_l = 0, temp_r = 0, suml = 0, sumr = 0;


    for (int j = line_start; j < line_end; j++)
    {
        for (int i = row_start; i < row_end; i++)
        {
            suml = 0.0, sumr = 0.0;
            temp_l = 0.0; temp_r = 0.0;
            for (int m = j - N; m <= j + N; m++)
            {
                for (int n = i - N; n <= i + N; n++)
                {
                    suml += ((uchar*)(left_img->imageData + m*left_img->widthStep))[n];
                    //cout << "l_px:" << (int)((uchar*)(left_img->imageData + m*left_img->widthStep))[n] << endl;
                    sumr += ((uchar*)(right_img->imageData + m*right_img->widthStep))[n];
                    //cout << "l_px:" << (int)((uchar*)(left_img->imageData + m*left_img->widthStep))[n] << endl;
                   sumr += ((uchar*)(right_img->imageData + m*right_img->widthStep))[n];
                   //cout << "r_px:" << (int)((uchar*)(right_img->imageData + m*right_img->widthStep))[n]<<endl;
               }
           }
           addr = j*width + i;
           left_avg[addr] = suml / pow((2 * N + 1), 2);
           right_avg[addr] = sumr / pow((2 * N + 1), 2);
           //cout << "l_avg:" << (float)left_avg[addr]<<endl;
           //cout << "r_avg:" << (float)right_avg[addr]<<endl;
           for (int m = j - N; m <= j + N; m++)
           {
               for (int n = i - N; n <= i + N; n++)
               {
                   temp_l += pow((((uchar*)(left_img->imageData + m*left_img->widthStep))[n] - left_avg[addr]), 2);
                   temp_r += pow((((uchar*)(right_img->imageData + m*right_img->widthStep))[n] - right_avg[addr]), 2);
               }


           }
           left_sq[addr] = temp_l;
           right_sq[addr] = temp_r;
           //cout << "l_sq:" << (float)left_sq[addr] << endl;
           //cout << "r_sq:" << (float)right_sq[addr] << endl;
       }


   }

}
