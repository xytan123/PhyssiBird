//
//  main.cpp
//  mp4
//
//  Created by Train on 10/25/13.
//  Copyright (c) 2013 Hao Wu. All rights reserved.
//


#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "opencv/highgui.h"
#include <cmath>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include "skinseg.h"
#include "fingertipdetect.h"
#include <time.h>

using namespace std;
using namespace cv;
#define CAMNUM 0
#define BACKGROUND_THRES 10
#define PI 3.14159265

static CvScalar colors[] =
{
    {{0,0,255}},
    {{0,128,255}},
    {{0,255,255}},
    {{0,255,0}},
    {{255,128,0}},
    {{255,255,0}},
    {{255,0,0}},
    {{255,0,255}}
};



void backgroundsubtraction(IplImage* frame_img, IplImage* background_img);
CvPoint center(IplImage* img);
IplImage* getthresholdedimg(IplImage* im);

int main(){
    int time_s, time_e;
    int key;
    //load object training image
    IplImage *train_img;//hand
    train_img = cvLoadImage("/Users/x/cpp/CAMERADETECT/training/hand.png");
    //train object
    skinseg skinsegmenter(train_img);
    skinsegmenter.training();
 
    //load t   arget training image
    IplImage *train_img2;//target
    train_img2 = cvLoadImage("/Users/x/cpp/CAMERADETECT/training/fixed.png");
    //train target
    skinseg skinsegmenter2(train_img2);
    skinsegmenter2.training();
    
    IplImage* bk=cvLoadImage("/Users/x/cpp/CAMERADETECT/training/bk.jpg");
//    Mat pig = pig_;
    
//    cout <<pig;
    
    
	CvCapture *capture = 0;
	capture = cvCaptureFromCAM( CAMNUM );
	
	if (!capture)
	{  
		printf("create camera capture error");
		system("pause");
		exit(-1);
	}

	CvSize size = cvSize((int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT));

    IplImage *frame, *frame_copy = 0;

    frame = cvQueryFrame( capture );
    IplImage *background_img = cvCreateImage( cvGetSize(frame), IPL_DEPTH_8U, 3 );
    cvCopy( frame, background_img);

    
    CvSeq* contours = 0;
    CvRect bound_rect;
    IplImage *grey_image = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
    IplImage *test = cvCreateImage(cvGetSize(frame),8,3);
    IplImage *image2 = cvCreateImage(cvGetSize(frame),8,3);
//    cvNamedWindow("Real",0);
//    cvNamedWindow("Threshold",0);
//    int i = 0;
    time_s = time(NULL);
    while((frame = cvQueryFrame(capture)) != NULL)
	{
        
		if (!frame_copy){
			frame_copy = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,frame->nChannels);
		}
		if (frame->origin == IPL_ORIGIN_TL){
			cvCopy (frame, frame_copy, 0);
		}
		else{
			cvFlip (frame, frame_copy, 0);
		}
        
        backgroundsubtraction(frame, background_img);
        cvShowImage("background", background_img);

//        //extra
////        IplImage * frame = cvQueryFrame(capture);
//        IplImage * imdraw = cvCreateImage(cvGetSize(frame),8,3);
//        cvSetZero(imdraw);
//        cvFlip(frame,frame,1);
//        cvSmooth(frame,frame,CV_GAUSSIAN,3,0);
//        IplImage * imgyellowthresh = getthresholdedimg(frame);
////        cout <<cvGetSize(imgyellowthresh).width<<" "<<cvGetSize(imgyellowthresh).height;
//        
//        
//        cvErode(imgyellowthresh,imgyellowthresh,NULL,3);
//        cvDilate(imgyellowthresh,imgyellowthresh,NULL,10);
//        cvShowImage("imgyell", imgyellowthresh);
//        IplImage * img2 = cvCloneImage(imgyellowthresh);
//        CvMemStorage* storage =cvCreateMemStorage(0);
////        cout <<contours;
//        cvFindContours(imgyellowthresh,storage,&contours,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
//
//        // CvPoint points[2] = {};
//        cout <<contours;
//        for(; contours!=0; contours = contours->h_next)
//        {
//            bound_rect = cvBoundingRect(contours, 0); //extract bounding box for current contour
//            cout <<bound_rect.x<<" "<<bound_rect.y<<" "<<bound_rect.x+bound_rect.width<<" "<<bound_rect.y+bound_rect.height;
//            //drawing rectangle
//            cvRectangle(frame_copy,
//                        cvPoint(bound_rect.x, bound_rect.y),
//                        cvPoint(bound_rect.x+bound_rect.width, bound_rect.y+bound_rect.height),
//                        CV_RGB(255,0,0),
//                        2);
//        }
//        
//        cvAdd(test,imdraw,test);
//        cvShowImage("Real",frame);
//        cvShowImage("Threshold",img2);
//        cvShowImage("Final",test);
//        
//        //end of extra
        
		IplImage *frame_resize = cvCreateImage(cvSize(frame->width/4, frame->height/4),IPL_DEPTH_8U,frame_copy->nChannels);
        cvResize(frame_copy, frame_resize);


        
        skinsegmenter.getskincolor(frame_copy);
        skinsegmenter2.getskincolor(frame_copy);

        IplImage *skin;
        skin = skinsegmenter.getskinimg();

        IplImage *skin2;
        skin2 = skinsegmenter2.getskinimg();
        
        IplImage *prediction = cvCreateImage(cvSize(bk->width,bk->height),IPL_DEPTH_8U,frame->nChannels);
        cvResize(prediction, frame_resize);
//        if(!bk) cout << "!";
//        if(!prediction) cout << "@@@";
        cvCopy(bk, prediction);
        
        
        
        CvPoint handcenter = center(skin);
        CvPoint targetcenter = center(skin2);
      
        cvCircle(frame_copy, targetcenter, 9, cvScalar(0, 255, 255), -1);
        cvCircle(frame_copy, handcenter, 9, cvScalar(0, 0, 255), -1);
        
        cout << "target center x: " << handcenter.x << " y: " << handcenter.y << endl;
        cout << "hand center x: " << targetcenter.x << " y: " << targetcenter.y << endl;
        
        //draw prediction
//        cvLine(prediction, handcenter,targetcenter, cvScalar(0, 0, 255), 3, 4);
        
        CvSize canvasSize =  cvGetSize(prediction);
        cout << "the width is "<< canvasSize.width<<endl<<"the height is "<< canvasSize.height;
        
        int A = abs(targetcenter.x - handcenter.x);
        int B = abs(targetcenter.y - handcenter.y);
        cout <<endl<< "A:"<<A<<" B:"<<B<<endl;
        int vx = A*0.1;
        int vy = B*0.1;
        int vy_wall;
//        int vx = 15;
//        int vy = 15;
//
        cout << "vx:"<<vx<<endl<<"vy:"<<vy;
        
        
        if(vx != 0){
            for(int i = 0; i < 50;i++){
                CvPoint to_draw;
                to_draw.x = i *(640/50);
                
//                cout <<"x" <<to_draw.x;
                int t = to_draw.x/vx;
//                cout <<"t"<<t;
//                cout <<"acc"<<2*pow(t,2)/2;
                to_draw.y = (240-vy*t + 0.8*pow(t,2)/2);
//                cout << "y"<<to_draw.y<<endl;
                cvCircle(prediction, to_draw, 5, cvScalar(9, 227, 38), -1);
                vy_wall = vy-0.8*t;
    //            cvLine(prediction, handcenter,targetcenter, cvScalar(0, 0, 255), 3, 4);
            }
        }
        
        CvFont font;
        double hScale=1.0;
        double vScale=1.0;
        int    lineWidth=2;
        string Force = "Force: ";
        string Angle = "Angle: ";
        
        
        
        string force_num = to_string(int(sqrt(pow(A,2)+pow(B, 2)))/10)+"N";
        Force = Force+force_num;
//        cout <<endl<<"the angle is "<<atan(B/A) * 180 / PI;
        
        string angle_num = to_string(atan(float(A)/float(B)) * 180 / PI);
//        double papram = 2/3;
        Angle = Angle+angle_num;
        int pr_y;
        if(vx!=0){
            cvLine(prediction, cvPoint(550,480),cvPoint(550,0), cvScalar(0, 0, 255), 3, 4);
            int t = 550/vx;
            pr_y = (240-vy*t + 0.8*pow(t,2)/2);
            //                cout << "y"<<to_draw.y<<endl;
            cvCircle(prediction, cvPoint(550, pr_y), 9, cvScalar(0, 255, 255), -1);
        
        
//            cvCircle(prediction, cvPoint(550, 400), 9, cvScalar(165, 206, 94), -1);
            
            cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
            
            
    //        cout << endl<<Force<<"<<"<<Angle<<endl;
            cvPutText (prediction,Force.c_str(),cvPoint(350,30), &font, cvScalar(0, 255, 255));
            
            cvPutText (prediction,Angle.c_str(),cvPoint(350,70), &font, cvScalar(0, 255, 255));
            
            string result;
            if (pr_y>480||pr_y<0) {
                result = "Out of range";
            }
            else if (abs(pr_y-400)<20){
                result = "Hit!";
                cvCircle(prediction, cvPoint(550, pr_y), 20, cvScalar(2,5,255), -1);
                
            }
            else {
                result = "Missed distance : " + to_string(400-pr_y);
                cvLine(prediction,cvPoint(550,pr_y),cvPoint(550,400),cvScalar(240,119,34),3,4);
            }
            
            cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale*0.7,vScale*0.7,0,lineWidth);
            
            cvPutText (prediction,result.c_str(),cvPoint(350,110), &font, cvScalar(0, 255, 255));
            
            string VX = "V_X: "+to_string(vx)+"m/s";
            string VY = "V_Y: "+to_string(vy)+"m/s";
            string VY_WALL = "V_Y HIT: "+to_string(vy_wall)+"m/s";
            cvPutText (prediction,VX.c_str(),cvPoint(350,140), &font, cvScalar(144, 139, 232));
            cvPutText (prediction,VY.c_str(),cvPoint(350,170), &font, cvScalar(144, 139, 232));
            cvPutText (prediction,VY_WALL.c_str(),cvPoint(350,210), &font, cvScalar(144, 139, 232));
        
        }
        
//        time_e = time(NULL);
//        int diff = difftime(time_e, time_s);
////        string time_mark = to_string(<#int __val#>);
//        cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 3.0,3.0,0,3);
//        cvPutText (prediction,(to_string(10-diff%11)).c_str(),cvPoint(10,70), &font, cvScalar(0, 0, 255));
//        cout << diff<<endl;
//        if (diff==11) {
//            
//            this_thread::sleep_for (chrono::seconds(5));
//            time_s = time(NULL);
//        }
        
        
        
        cout <<endl<<"============"<<endl;
        cvShowImage("skin", skin);
        cvMoveWindow("skin", 1200, 0);
        
        cvShowImage("skin2", skin2);
        cvMoveWindow("skin2", 1200, 550);
        
        cvShowImage("frame", frame_copy);
        cvMoveWindow("frame", 650, 0);
        
        cvShowImage("prediction", prediction);
        cvMoveWindow("prediction", 0, 0);
        
//        cvShowImage("prediction", train_img2);
        
        
        
        key = cvWaitKey( 1 );
        if (key == 'q') {
            // 'q'
            break;
        }
        else if (key == 'u') {
            // 'u' update
            frame = cvQueryFrame( capture );
            cvCopy( frame, background_img );
        }
        else if (key == 's'){
            cvSaveImage("/users/shana/desktop/frame.bmp", frame);
            //            skinsegmenter.training(frame);
        }
        
//        i++;
        
    
    
    }
    
	return 0;
    
}


void backgroundsubtraction(IplImage* frame_img, IplImage* background_img){
    IplImage *grayImage = cvCreateImage( cvGetSize(frame_img), IPL_DEPTH_8U, 3 );
    IplImage *differenceImage = cvCreateImage( cvGetSize(frame_img), IPL_DEPTH_8U, 3 );
    IplImage *differenceImage_gray = cvCreateImage( cvGetSize(frame_img), IPL_DEPTH_8U, 1 );
    cvCopy( frame_img, grayImage);
    cvSmooth(grayImage, grayImage, CV_GAUSSIAN, 3, 0, 0);
    
    cvAbsDiff( grayImage, background_img, differenceImage );
    cvCvtColor(differenceImage, differenceImage_gray, CV_BGR2GRAY);
    cvThreshold( differenceImage_gray, differenceImage_gray, BACKGROUND_THRES, 255, CV_THRESH_BINARY );

    cvErode(differenceImage_gray,differenceImage_gray,NULL,1);
    cvDilate(differenceImage_gray,differenceImage_gray,NULL,1);
    //        cvMorphologyEx(differenceImage_gray,differenceImage_gray,NULL,kernel_ellipse,CV_MOP_CLOSE,1);
    cvSmooth(differenceImage_gray, differenceImage_gray, CV_GAUSSIAN, 5, 0, 0);
    
    Mat frame_mat = frame_img;
    Mat mask_mat = differenceImage_gray;
    
    for (int i = 0; i < mask_mat.rows; i++){
        uchar * Mi = frame_mat.ptr(i);
        for (int j = 0; j < mask_mat.cols; j++){
            if (mask_mat.at<uchar>(i, j) == 0) {
                frame_mat.at<uchar>(i, 3*j) = 0;
                frame_mat.at<uchar>(i, 3*j+1) = 0;
                frame_mat.at<uchar>(i, 3*j+2) = 0;
            }
        }
    }
    //    cvShowImage( "Difference", differenceImage );
//    cvShowImage("differenceImage_gray", differenceImage_gray);
}

CvPoint center(IplImage* img){
    CvPoint center;
    CvPoint zero;
    zero.x = 0;
    zero.y = 0;
    double m00, m10, m01;
    CvMoments moment;
    cvMoments( img, &moment, 1);
    m00 = cvGetSpatialMoment( &moment, 0, 0 );
    if( m00 == 0)
        return zero;
    m10 = cvGetSpatialMoment( &moment, 1, 0 );
    m01 = cvGetSpatialMoment( &moment, 0, 1 );
    center.x = (int) (m10/m00);
    center.y = (int) (m01/m00);
    return center;
}

IplImage* getthresholdedimg(IplImage* im){
	IplImage * imghsv= cvCreateImage(cvGetSize(im),8,3);
	cvCvtColor(im,imghsv,CV_BGR2HSV);
    
	IplImage * imgpink = cvCreateImage(cvGetSize(im),8,1);
//	IplImage * imgblue = cvCreateImage(cvGetSize(im),8,1);
	IplImage * imggreen = cvCreateImage(cvGetSize(im),8,1);
    
	IplImage *imgthreshold = cvCreateImage(cvGetSize(im),8,1);
    
	cvInRangeS(imghsv,cvScalar(201,247,181),cvScalar(10,94,2),imggreen);
	cvInRangeS(imghsv,cvScalar(219,187,250),cvScalar(41,0,247),imgpink);
//	cvInRangeS(imghsv,cvScalar(100,100,100),cvScalar(120,255,255),imgblue);
//    cvShowImage("green", imggreen);
//    cvShowImage("blue", imgblue);
//    cvShowImage("yell", imgyellow);
	cvAdd(imgpink,imggreen,imgthreshold);
//	cvAdd(imgthreshold,imgthreshold);
	return imgthreshold;
}
