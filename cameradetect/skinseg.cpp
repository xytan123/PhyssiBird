//
//  skinseg.cpp
//  fingerdetect
//
//  Created by Train on 11/26/13.
//  Copyright (c) 2013 Hao Wu. All rights reserved.
//

#include "skinseg.h"

skinseg::skinseg(IplImage *training_img){
    training_hsv = cvCreateImage(cvGetSize(training_img),8,3);
    cvCvtColor(training_img,training_hsv,CV_BGR2HSV);
}

skinseg::~skinseg(){
    
}

void skinseg::getskincolor(IplImage *img){
    IplImage *test_img = cvCreateImage(cvGetSize(img),8,3);
    cvCopy(img, test_img);
    skin_img = cvCreateImage(cvGetSize(img),8,1);
    
    IplImage *test_img_hsv = 0;
    test_img_hsv = cvCreateImage(cvGetSize(test_img),8,3);
    cvCvtColor(test_img,test_img_hsv,CV_BGR2HSV);
    
    Color(test_img_hsv, histo_mat_hsv);
    
    IplImage * hsv_result = 0;
    hsv_result = cvCreateImage(cvGetSize(test_img),8,3);
    cvCvtColor(test_img_hsv,hsv_result,CV_HSV2BGR);
    
    
    IplImage* skin2;
    skin2 = cvCreateImage(cvGetSize(test_img),8,1);
    cvCvtColor(hsv_result,skin2,CV_BGR2GRAY);
    
    
    IplImage* skin_dilate;
    skin_dilate = cvCreateImage(cvGetSize(test_img),8,1);
    
    IplImage* skin_erode;
    skin_erode = cvCreateImage(cvGetSize(test_img),8,1);
    
    IplImage* skin_close;
    skin_close = cvCreateImage(cvGetSize(test_img),8,1);
    
    IplConvKernel *kernel_ellipse = cvCreateStructuringElementEx(9,9,4,4,CV_SHAPE_ELLIPSE,NULL);
    cvErode(skin2,skin_erode,NULL,1);
    cvDilate(skin2,skin_dilate,kernel_ellipse,1);
    cvMorphologyEx(skin_dilate,skin_close,NULL,kernel_ellipse,CV_MOP_CLOSE,1);
    
    IplImage* skin_label;
    skin_label = cvCreateImage(cvGetSize(test_img),8,1);
    cvCopy(skin_close, skin_label);
    
    Mat skin_label_mat = skin_label;
    
    
    Mat binary;
    threshold(skin_label_mat, binary, 0.0, 1.0, cv::THRESH_BINARY);
    
    std::vector < vector<Point2i> > blobs;
    FindBlobs(binary, blobs);
    
    
    int max_blobs_number = 0;
    for(int i=0; i < blobs.size(); i++) {
        if (blobs[i].size() > blobs[max_blobs_number].size()) {
            max_blobs_number = i;
        }
        
    }
    for (int i = 0; i < skin_label_mat.cols; i++) {
        for (int j = 0; j < skin_label_mat.rows; j++) {
            skin_label_mat.at<uchar>(j, i) = 0;
        }
    }

    
    for (int i= 0; i < skin_label_mat.cols; i++) {
        for (int j = 0; j < skin_label_mat.rows; j++) {
            skin_label_mat.at<uchar>(j,i) = 0;
        }
    }
    
    if (blobs.size() != 0) {
        for(int i=0; i < blobs[max_blobs_number].size(); i++) {
            int x = blobs[max_blobs_number][i].x;
            int y = blobs[max_blobs_number][i].y;
            skin_label_mat.at<uchar>(y,x) = 255;
        }
    }
    IplImage* skin_smooth;
    skin_smooth = cvCreateImage(cvGetSize(test_img),8,1);
    cvSmooth(skin_label, skin_smooth, CV_GAUSSIAN, 5, 0, 0);
    
    cvCopy(skin_smooth, skin_img);

}

void skinseg::training(){
    Mat training_hsv_mat = training_hsv;
    histo_mat_hsv = Histo(training_hsv_mat);
}

void skinseg::training(IplImage* training_img){
    training_hsv = cvCreateImage(cvGetSize(training_img),8,3);
    cvCvtColor(training_img,training_hsv,CV_BGR2HSV);
    Mat training_hsv_mat = training_hsv;
    histo_mat_hsv = Histo(training_hsv_mat);
}

Mat skinseg::Histo(Mat training_mat){
    int col = training_mat.cols;
    int row = training_mat.rows;
    Mat HShisto = Mat::zeros(256, 256, CV_8UC1);
    for (int i = 0; i < row; i++){
        uchar * Mi = training_mat.ptr(i);
        for (int j = 0; j < col; j++){
            int H = (int)Mi[3*j];
            int S = (int)Mi[3*j+1];
            HShisto.at<int>(H,S)++;
        }
    }
    
    
    Mat HShisto_normalize = Mat::zeros(256, 256, CV_32FC1);
    int pixels = col*row;
    for (int i = 0; i < 256; i++){
        for (int j = 0; j < 256; j++){
            HShisto_normalize.at<float>(i,j) = (float)HShisto.at<int>(i,j)/pixels;
        }
    }
    return HShisto_normalize;
}

void skinseg::colorNormal(Mat& img)
{
    int nl= img.rows; // number of lines
    int nc= img.cols ; // number of columns
    // is it a continous image?
    if (img.isContinuous())
    {
        // then no padded pixels
        nc= nc*nl;
        nl= 1;  // it is now a 1D array
    }
    
    double bgrSum;
    // for all pixels
    for (int j=0; j<nl; j++)
    {
        // pointer to first column of line j
        uchar* data= img.ptr(j);
        for (int i=0; i<nc; i++)
        {
            // process each pixel --------
            bgrSum= 255.0/(0.0000000001 +data[0] +data[1] +data[2]);
            data[0] *=bgrSum;
            data[1] *=bgrSum;
            data[2] *=bgrSum;
            data +=3;
        }
    }
}

void skinseg::Color(IplImage *img, Mat histo)
{
    
	int i,j;
    
	struct num **bmpdata;
	struct num **bmpdata1;
	bmpdata = new num*[img->height];
	bmpdata1 = new num*[img->height];
    
	for(i=0;i<img->height;i++)
	{
		bmpdata[i] = new num[img->width];
		bmpdata1[i] = new num[img->width];
	}
	
    
	for(i=0;i<img->height;i++)
		for(j=0;j<img->width;j++)
		{
			bmpdata[i][j].H=((uchar*)(img->imageData + img->widthStep*i))[j*3];
		    bmpdata[i][j].S=((uchar*)(img->imageData + img->widthStep*i))[j*3+1] ;
			bmpdata[i][j].V=((uchar*)(img->imageData + img->widthStep*i))[j*3+2];
		}
    
    
	for (i=0;i<img->height;i++)
	    for (j=0;j<img->width;j++)
		{
            int H = (int)bmpdata[i][j].H;
            int S = (int)bmpdata[i][j].S;
            int V = (int)bmpdata[i][j].V;
//            if(bmpdata[i][j].H > 25 || bmpdata[i][j].S < 26 || bmpdata[i][j].V < 20)
            if(histo.at<float>(H,S) < THRESHOLD)
            {
                bmpdata[i][j].H = bmpdata[i][j].S = 120;
                bmpdata[i][j].V = 0;
            }
            else{
                bmpdata[i][j].H = bmpdata[i][j].S = 0;
                bmpdata[i][j].V = 255;
            }
		}
	
    
    
    
	for (i=0;i<img->height;i++)
 		for (j=0;j<img->width;j++)
		{
		    ((uchar*)(img->imageData + img->widthStep*i))[j*3]=bmpdata[i][j].H;
            
            ((uchar*)(img->imageData + img->widthStep*i))[j*3+1]=bmpdata[i][j].S;
	    	((uchar*)(img->imageData + img->widthStep*i))[j*3+2]=bmpdata[i][j].V;
	    }
    
}

void skinseg::FindBlobs(const Mat &binary, vector < vector<Point2i> > &blobs)
{
    blobs.clear();
    
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    
    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    
    int label_count = 2; // starts at 2 because 0,1 are used already
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }
            
            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);
            
            vector <Point2i> blob;
            
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    
                    blob.push_back(Point2i(j,i));
                }
            }
            
            blobs.push_back(blob);            
            label_count++;
        }
    }
}