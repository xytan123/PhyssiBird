//
//  fingertipdetect.h
//  fingertip
//
//  Created by Train on 11/27/13.
//  Copyright (c) 2013 Hao Wu. All rights reserved.
//

#ifndef __fingertip__fingertipdetect__
#define __fingertip__fingertipdetect__

#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "opencv/highgui.h"
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;
using namespace cv;


class fingertipdetect {
private:
	IplImage* pImgGray;
	IplImage* pImgContourAll;
	IplImage* pImgContourAppr;
	IplImage* pImgHull;
	IplImage* pImgDefects;
    CvPoint contourcenter;
    CvPoint max_hull;
    CvPoint min_defect;
public:
    fingertipdetect(IplImage* img);
    ~fingertipdetect();
    void tipdetect();
    float distance(CvPoint, CvPoint);
    float cos(CvConvexityDefect*);
    CvPoint center(IplImage*);
    void max_h(CvSeq* hull);
    void min_d(CvSeq* defect);
    void rearrange(CvSeq* defectSeq, CvSeq* hullp);
    void rearrange2(CvSeq* defectSeq);
    CvSeq* hulls;
    CvSeq* defects;
    CvSeq* contourapprox;
    CvSeq* contour;
    void draw();
    void show();
    CvPoint get_max(){
        return max_hull;
    }
    CvPoint get_min(){
        return min_defect;
    }
    CvPoint get_center(){
        return contourcenter;
    }

};





#endif /* defined(__fingertip__fingertipdetect__) */
