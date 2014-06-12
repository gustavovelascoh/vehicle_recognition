/*
 * Bgsub.h
 *
 *  Created on: Jun 12, 2014
 *      Author: gustavo
 */

#ifndef BGSUB_H_
#define BGSUB_H_

#include "opencv2/opencv.hpp"
//#include "opencv2/video/video.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class Bg_sub : public BackgroundSubtractorMOG2
{
public:
	float fTau;
	Bg_sub(int history,  float varThreshold, bool bShadowDetection=true, float shT=0.1): BackgroundSubtractorMOG2(history, varThreshold, bShadowDetection) {
				fTau = shT;
	}
	//Bg_sub();
	//virtual ~Bg_sub();
};

#endif /* BGSUB_H_ */
