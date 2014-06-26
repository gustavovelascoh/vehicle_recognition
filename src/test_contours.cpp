/*
 * test_contours.cpp
 *
 *  Created on: Jun 22, 2014
 *      Author: gustavo
 */
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

RNG rng(12345);

int main (int argc, char** argv)
{
	/* ---- Init code ---- */
	if (argc == 1 )
		return -1;

	string input = argv[1];
	VideoCapture cap = VideoCapture(input);
	cout << "Opening: " << input << " -> ";

	if(!cap.isOpened())  // check if we succeeded
	{
		cout << "ERROR" << endl;
		return -1;
	}
	cout << "OK" << endl;
	/* ---- Init code ---- */
	namedWindow("Input",1);
	namedWindow("Output",1);

	for(;;)
	{
		Mat frame;

		vector<vector<Point> > contours;
		vector<vector<Point> > contours_r;
		Mat frame_gray, threshold_output;
		vector<Vec4i> hierarchy;
		cap >> frame; // get a new frame from camera
		imshow("Input", frame );


		cvtColor( frame, frame_gray, CV_BGR2GRAY );

		/// Detect edges using Threshold
		threshold( frame_gray, threshold_output, 64, 255, THRESH_BINARY );

		/// Find contours
		findContours( threshold_output, contours_r, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		cout << "contours #: " << contours_r.size() << endl;

		for( int i = 0; i < contours_r.size(); i++ )
		{
			cout << "Contour " << i << " size: " << contours_r[i].size() << ".";
			cout << "(" << contours_r[i][0].x << "," << contours_r[i][0].y << ")" << endl;

			if (contours_r[i].size() > 80)
			{
				contours.push_back(contours_r[i]);
			}

		}

		waitKey(0);

		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );

		for( int i = 0; i < contours.size(); i++ )
		{
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		//minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
		}
		cout << "cont size: " << contours.size() << endl;
		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
		}


		imshow("Output", drawing );

		//waitKey(0);
			  //return(0);


		if(waitKey(30) >= 0) break;

		//return 0;
	}

	return 0;

}


