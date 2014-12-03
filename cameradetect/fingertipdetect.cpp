//
//  fingertipdetect.cpp
//  fingertip
//
//  Created by Train on 11/27/13.
//  Copyright (c) 2013 Hao Wu. All rights reserved.
//

#include "fingertipdetect.h"
#define THRES_APPR 0.0001
fingertipdetect::fingertipdetect(IplImage* img){
    
    pImgGray=cvCreateImage(cvGetSize(img), 8, 1);
    cvCopy(img,pImgGray);
	pImgContourAppr = cvCreateImage(cvGetSize(pImgGray), 8, 3);
	pImgContourAll = cvCreateImage(cvGetSize(pImgGray), 8, 3);
	pImgHull = cvCreateImage(cvGetSize(pImgGray), 8, 3);
	pImgDefects = cvCreateImage(cvGetSize(pImgGray), 8, 3);
	cvZero(pImgContourAppr);
	cvZero(pImgContourAll);
	cvZero(pImgHull);
	cvZero(pImgDefects);
    contourcenter = center(pImgGray);
    CvPoint zero;
    zero.x = 0;
    zero.y = 0;
    max_hull = contourcenter;
    min_defect = zero;
}

fingertipdetect::~fingertipdetect(){
    
}

float fingertipdetect::distance(CvPoint p1, CvPoint p2){
    return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

float fingertipdetect::cos(CvConvexityDefect* p){
    float x, y, z, cos;
    x = distance(*p->depth_point, *p->start);
    y = distance(*p->depth_point, *p->end);
    z = distance(*p->start, *p->end);
    cos = (x*x + y*y - z*z)/(2*x*y);
    return cos;
}

CvPoint fingertipdetect::center(IplImage* img){
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

void fingertipdetect::max_h(CvSeq* hull){
    for (int i=0; i < hull->total; i++){
        CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, hull, i);
        if(distance(contourcenter, max_hull) < distance(contourcenter, *p)){
            max_hull = *p;
        }
    }
    //    cout << "max hull: " << max_hull.x << ", " << max_hull.y << endl;
    //    cvCircle(pImgDefects, max_hull, 9, cvScalar(200, 0, 200));
    //    cvCircle(pImgHull, max_hull, 9, cvScalar(200, 0, 200));
    
}

void fingertipdetect::min_d(CvSeq* defect){
    for (int i=0; i < defect->total; i++){
        CvConvexityDefect* p = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defect, i);
        if(distance(contourcenter, min_defect) > distance(contourcenter, *p->depth_point)){
            min_defect = *p->depth_point;
        }
    }
    //    cout << "min defect: " << min_defect.x << ", " << min_defect.y << endl;
    //    cvCircle(pImgDefects, min_defect, 10, cvScalar(120, 120, 120));
    //    cvCircle(pImgHull, min_defect, 10, cvScalar(120, 120, 120));
}

void fingertipdetect::rearrange(CvSeq* defectSeq, CvSeq* hullp){
    int i = 0;
    while(i < defectSeq->total){
    	CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defectSeq, i);
        for (int j=0; j < hullp->total; j++){
            CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, hullp, j);
            if (dp->depth_point->x == p->x && dp->depth_point->y == p->y)
                //            if (distance(*dp->depth_point, *p) < 20)
            {
                //                cout << "remove1 distance: " << dp->depth_point->x << ", " << dp->depth_point->y << " " << defectSeq->total << endl;
                cvSeqRemove(defectSeq, i);
                i--;
                break;
            }
        }
        i++;
    }
    i = 0;
    while(i < defectSeq->total){
    	CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defectSeq, i);
        if (cos(dp) < -0.1 || cos(dp) > 0.985) {
            //            cout << "remove2 angle: " << dp->depth_point->x << ", " << dp->depth_point->y << " " << defectSeq->total << endl;
            cvSeqRemove(defectSeq, i);
            i--;
        }
        i++;
    }
    return;
    
}

void fingertipdetect::rearrange2(CvSeq* defectSeq){
    int i = 0;
    float distance_threshold1, distance_threshold2;
    float percentage1 = 0.7;
    float percentage2 = 0.9;
    
    distance_threshold1 = percentage1 * (distance(contourcenter, max_hull) + distance(contourcenter, min_defect));
    distance_threshold2 = percentage2 * distance(contourcenter, max_hull);
    
    while(i < defectSeq->total){
    	CvConvexityDefect* p = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defectSeq, i);
        if (distance(contourcenter, *p->depth_point) > distance_threshold1 ||
            distance(contourcenter, *p->depth_point) > distance_threshold2) {
            //            cout << "remove3 distance2: " << p->depth_point->x << ", " << p->depth_point->y << " " << defectSeq->total << endl;
            cvSeqRemove(defectSeq, i);
            i--;
        }
        i++;
    }
}


void fingertipdetect::tipdetect(){
    //canny
    CvMemStorage* storage = cvCreateMemStorage();
    CvSeq* contourSeqAll =cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
    cvSmooth(pImgGray, pImgGray, CV_GAUSSIAN);
    cvCanny(pImgGray, pImgGray, 10, 30, 5);
    cvFindContours(pImgGray, storage, &contourSeqAll, sizeof(CvContour), CV_RETR_LIST, CV_LINK_RUNS);
    //original contours
    CvSeq* tseq = contourSeqAll;
    for (; contourSeqAll; contourSeqAll = contourSeqAll->h_next){
        cvDrawContours(pImgDefects, contourSeqAll, cvScalar(255,0,0), cvScalar(0,0,255), 0, 2);
    }
    contourSeqAll = tseq;
    contour = contourSeqAll;
    CvMemStorage* storageAppr = cvCreateMemStorage();
    CvSeq* contourAppr = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storageAppr);
    if (contourSeqAll == NULL) {
        cout << "no object" << endl;
        return;
        //        exit(0);
    }
    contourAppr = cvApproxPoly(contourSeqAll, sizeof(CvContour), storageAppr, CV_POLY_APPROX_DP, cvContourPerimeter(contourSeqAll)*THRES_APPR, 1);
    //approximated contours
    tseq = contourAppr;
    for (; contourAppr; contourAppr = contourAppr->h_next){
        cvDrawContours(pImgContourAppr, contourAppr, cvScalar(0,255
                                                              ,0), cvScalar(0,0,255), 0, 1);
    }
    contourAppr = tseq;
    contourapprox = contourAppr;
//    cvShowImage("contourappr", pImgContourAppr);
    //print contours
    //	cout<<"contours:"<<endl;
    //    for (int i=0;i<contourAppr->total;i++){
    //        CvPoint* p=CV_GET_SEQ_ELEM(CvPoint,contourAppr,i);
    //        cout<<p->x<<","<<p->y<<endl;
    //        cvCircle(pImgHull,*p,3,cvScalar(0,255,255));
    //        cvShowImage("hull",pImgHull);
    //        cvWaitKey(0);
    //     }
    
    
    cvCircle(pImgDefects, contourcenter, 7, cvScalar(100, 255, 200));
//    cvShowImage("contourall", pImgContourAll);
    
    
    
    
    ////convex hull
    CvSeq* hull = cvConvexHull2(contourAppr);
    CvSeq* hullp = cvConvexHull2(contourAppr, 0, 0, 1 );
    hulls = hullp;
    
    //    cout<<"hull: "<<endl;
    for (int i=0; i < hullp->total; i++){
        CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, hullp, i);
        //        cout<< p->x << "," << p->y << endl;
        cvCircle(pImgDefects, *p, 5, cvScalar(0, 255, 0));
        //        cvShowImage("hull", pImgHull);
        //        cvWaitKey(0);
    }
    
    
    //convexity defects
    CvSeq* defectSeq = cvConvexityDefects(contourAppr, hull);
    
    
    
    
    //rearrange the detectSeq in linked sequence
    rearrange(defectSeq, hullp);
    
    
    
    
    max_h(hullp);
    min_d(defectSeq);
    
    
    rearrange2(defectSeq);
    
    
    defects = defectSeq;
    
    //    for (int i=0; i<defects->total; i++){
    //        CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, i);
    //
    //        cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
    //        //        cvShowImage("ConvexityDefects", pImgDefects);
    //        //        cvWaitKey(0);
    //        cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
    //        //        cvShowImage("ConvexityDefects", pImgDefects);
    //        //        cvWaitKey(0);
    //        cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
    //        //        cvShowImage("ConvexityDefects", pImgDefects);
    //        //        cvWaitKey(0);
    //        cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
    //        //        cvWaitKey(0);
    //
    //        //        cout<< i << " defect :(" << dp->depth_point->x << "," << dp->depth_point->y << ")" << endl;
    //
    //        //        cout<< i << " defect :(" << dp->start->x << "," << dp->start->y << ")" << endl;
    //        //        if (i == defectSeq->total-1) {
    //        //            cout<< i+1 << " defect :(" << dp->end->x << "," << dp->end->y << ")" << endl;
    //        //        }
    //    }
    
    draw();
//    show(); 
    
    //	cvShowImage("original", pImgColor);
    //	cvShowImage("canny", pImgGray);
//    cvShowImage("contourappr", pImgContourAppr);
    cvShowImage("ConvexityDefects", pImgDefects);
    cvShowImage("hull", pImgHull);
    
    
//    cvMoveWindow("contourappr", 30, 100);
//	cvMoveWindow("ConvexityDefects", 360, 0);
//	cvMoveWindow("hull", 500, 280);
    
    //    cvWaitKey(0);
    
}

void fingertipdetect::draw(){
    if (defects->total == 0) {
        cvCircle(pImgHull, max_hull, 9, cvScalar(200, 0, 200), -1);
        cvCircle(pImgDefects, max_hull, 9, cvScalar(200, 0, 200), -1);
    }
    
    else if(defects->total == 1){
        CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 0);
        
        cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
        
        //        cout<< i << " defect :(" << dp->depth_point->x << "," << dp->depth_point->y << ")" << endl;
        //        cout<< i << " defect :(" << dp->start->x << "," << dp->start->y << ")" << endl;
    }
    
    else if(defects->total == 2){
        CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 0);
        
        cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
        
        dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 1);
        cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
        //        cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
        //        cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
    }
    
    else if(defects->total == 3){
        CvConvexityDefect* dp1 = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 0);
        
        cvCircle(pImgDefects, *(dp1->depth_point), 3, cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp1->start), *(dp1->end), cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp1->start), *(dp1->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp1->depth_point), *(dp1->end), cvScalar(0xff,0x99,0x00));
        
        CvConvexityDefect* dp2 = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 2);
        
        cvCircle(pImgDefects, *(dp2->depth_point), 3, cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp2->start), *(dp2->end), cvScalar(0,0,255));
        cvLine(pImgDefects, *(dp2->start), *(dp2->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp2->depth_point), *(dp2->end), cvScalar(0xff,0x99,0x00));
        
    }
    
    else if(defects->total == 4){
        for (int i=0; i<defects->total; i = i + 2){
            CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, i);
            cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
            cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
            cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
            cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
        }
        CvConvexityDefect* dp = (CvConvexityDefect*)CV_GET_SEQ_ELEM(CvConvexityDefect, defects, 3);
        cvCircle(pImgDefects, *(dp->depth_point), 3, cvScalar(0,0,255));
        //        cvLine(pImgDefects, *(dp->start), *(dp->end), cvScalar(0,0,255));
        //        cvLine(pImgDefects, *(dp->start), *(dp->depth_point), cvScalar(0x00,0x99,0xff));
        cvLine(pImgDefects, *(dp->depth_point), *(dp->end), cvScalar(0xff,0x99,0x00));
        
    }
    else {
        cvCircle(pImgHull, max_hull, 9, cvScalar(100, 100, 200), -1);
        cvCircle(pImgDefects, max_hull, 9, cvScalar(100, 100, 200), -1);
    }
    
    cvCircle(pImgDefects, min_defect, 10, cvScalar(120, 120, 120), -1);
    cvCircle(pImgHull, min_defect, 10, cvScalar(120, 120, 120), -1);
    
    return;
}

void fingertipdetect::show(){
    
    
    IplImage* one = cvLoadImage("/users/shana/desktop/gesture/1.jpeg");
    IplImage* two = cvLoadImage("/users/shana/desktop/gesture/2.jpeg");
    IplImage* three = cvLoadImage("/users/shana/desktop/gesture/3.jpeg");
    IplImage* four = cvLoadImage("/users/shana/desktop/gesture/4.jpeg");
    IplImage* five = cvLoadImage("/users/shana/desktop/gesture/5.jpeg");
    IplImage* ten = cvLoadImage("/users/shana/desktop/gesture/10.jpeg");
    switch (defects->total) {
        case 0:
            cvShowImage("gesture", one);
            break;
        case 1:
            cvShowImage("gesture", two);
            
            break;
        case 2:
            cvShowImage("gesture", three);
            
            break;
        case 3:
            cvShowImage("gesture", four);
            break;
        case 4:
            cvShowImage("gesture", five);
            cvMoveWindow("gesture", max_hull.x,  max_hull.y);
            break;
            
        default:
            cvShowImage("gesture", ten);
            
            break;
    }
}