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


}


