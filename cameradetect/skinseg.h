//
//  skinseg.h
//  fingerdetect
//
//  Created by Train on 11/26/13.
//  Copyright (c) 2013 Hao Wu. All rights reserved.
//

#ifndef __fingerdetect__skinseg__
#define __fingerdetect__skinseg__

#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include"opencv/highgui.h"
#include <vector>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
#define THRESHOLD 0.000000000000000000001

struct num
{
    unsigned char H;
    unsigned char S;
    unsigned char V;
};

class skinseg {
private:
    IplImage *training_hsv;
    Mat histo_mat_hsv;
    IplImage *skin_img;
public:
    
    skinseg(IplImage *train_img);
    ~skinseg();
    void training();
    void getskincolor(IplImage *img);
    void colorNormal(Mat& img);
    void Color(IplImage *img, Mat histo);
    Mat Histo(Mat training_mat);
    IplImage *getskinimg(){
        return skin_img;
    }
    void training(IplImage* training_img);

    void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);
};











#endif /* defined(__fingerdetect__skinseg__) */
