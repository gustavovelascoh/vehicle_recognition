#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
void show_cap_info(VideoCapture cap, std::string);
void morph_ops(InputArray src, OutputArray dst);

int main(int argc, char** argv)
{
    int key = 0, crop = 0;

    VideoCapture cap;
    std::string input;

    if (argc < 2)
    {
        input = "./media_src/the_video.mp4";
        crop = 1;
    }
    else
    {
        input = argv[1];
    }
    
    cap = VideoCapture(input);

    if(!cap.isOpened())  // check if we succeeded
    {
    	std::cout << "Error opening " << input << std::endl;
    	return -1;
    }

    show_cap_info(cap,input);

    //static const int arr[] = {2160, 2260, 4380, 6170, 6390};
    static const int arr[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000};
    std::vector<int> frames_to_save (arr, arr + sizeof(arr) / sizeof(arr[0]) );
    int save_frames = 1;
    
    int history =  500;
    float varThreshold = 9;
    
    std::stringstream ln1, ln2, ln3;
    
    BackgroundSubtractorMOG2 bg_sub(history,varThreshold,false);
    
    //Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(2,2));
    //namedWindow("Frame",1);
    //namedWindow("Mask",1);
    //namedWindow("Closing(Frame & Mask)",1);
    double t = (double)getTickCount();
    for(;;)
    {
        Mat frame, bg_mask, frame_masked, cl_mask, op_mask;
        cap >> frame; // get a new frame from camera
        
        if (crop)
        	frame = frame.colRange(160,1120);


        GaussianBlur(frame, frame, Size(5,5), 1.5, 1.5);
        
        // Execute background subtraction and get mask
        bg_sub(frame,bg_mask);
        // Apply opening + closing operation to the mask

        morph_ops(bg_mask, bg_mask);
        //morphologyEx(bg_mask, bg_mask, CV_MOP_OPEN, element, Point(2,2),2);
        //morphologyEx(bg_mask, bg_mask, CV_MOP_CLOSE, element, Point(2,2), 5);
        //morphologyEx(cl_mask, op_mask, CV_MOP_OPEN, element, Point(2,2), 3);
        // mask original frame
        frame.copyTo(frame_masked,bg_mask);
        

        int curr_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
        std::stringstream ss;
        ss << "frame " << curr_frame;
        std::string fns = ss.str();
		
        if (curr_frame > 7001)
        {
        	t = ((double)getTickCount() - t)/getTickFrequency();

        	std::cout << "Time elapsed: " << t << " secs" << std::endl;
        	return 1;
        }
        /*
        rectangle(frame,cv::Point(10,5),cv::Point(120,25),cv::Scalar(255,255,255,-1),CV_FILLED);
		rectangle(bg_mask,cv::Point(10,5),cv::Point(120,25),cv::Scalar(255,255,255,-1),CV_FILLED);
		rectangle(frame_masked,cv::Point(10,5),cv::Point(120,25),cv::Scalar(255,255,255,-1),CV_FILLED);

        putText(frame,fns.c_str(),cv::Point(25,20),FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
		putText(bg_mask,fns.c_str(),cv::Point(25,20),FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
		putText(frame_masked,fns.c_str(),cv::Point(25,20),FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
		*/

        //bgsub1.getBackgroundImage(bg1);
		

		//std::cout << fns << std::endl;

		// Save frames
        if (save_frames)
        {
			if(std::find(frames_to_save.begin(), frames_to_save.end(), curr_frame)!=frames_to_save.end())
			{
				std::cout << fns << std::endl;
				std::stringstream ln1, ln2, ln3, com;

				ln1 << "f" << curr_frame;
				//ln2 << ln1.str();
				//ln3 << ln1.str();

				//com << "h" << history << "v" << varThreshold << "oc2.jpg";
				ln1 << "fm3.jpg";

				//ln1 << ".jpg";
				//ln2 << "_b" << com.str();
				//ln3 << "_m" << com.str();

				imwrite(ln1.str().c_str(),frame_masked);
				//imwrite(ln2.str().c_str(),bg_mask);
				//imwrite(ln3.str().c_str(),frame_masked);
			}
        }
        //imshow("Frame", frame);
        //imshow("Mask", bg_mask);
        //imshow("Closing(Frame & Mask)", frame_masked);
        
		// wait key statements

        key = (char) waitKey(5);
        //std::cout << "key " << key << std::endl;
        if (key == 112)
        {
            while(key < 0 || key == 112)
            {
                key = waitKey(5);
            }
        }        
        else if(key >= 0)
        {
            break;
        }

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

void show_cap_info(VideoCapture cap, std::string input)
{
	Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
	                      (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	        std::cout << "Opening file " << input << std::endl;
	        std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
	             << " of nr#: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
}

void morph_ops(InputArray src, OutputArray dst)
{
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(2,2));
	Mat temp;

	morphologyEx(src, temp, CV_MOP_OPEN, element, Point(2,2),2);
	morphologyEx(temp, dst, CV_MOP_CLOSE, element, Point(2,2), 10);
}

/// Separate the image in 3 places ( B, G and R )
		//vector<Mat> bgr_planes, bgr_eq;
		//split(frame, bgr_planes );

		/*
		equalizeHist(bgr_planes[0],bgr_planes[0]);
		equalizeHist(bgr_planes[1],bgr_planes[1]);
		equalizeHist(bgr_planes[2],bgr_planes[2]);

		merge(bgr_planes, frame);

		/// Establish the number of bins
		  int histSize = 256;

		  /// Set the ranges ( for B,G,R) )
		  float range[] = { 0, 256 } ;
		  const float* histRange = { range };

		  bool uniform = true; bool accumulate = false;

		  Mat b_hist, g_hist, r_hist;

		  /// Compute the histograms:
		  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
		  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
		  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

		  // Draw the histograms for B, G and R
		  int hist_w = 512; int hist_h = 400;
		  int bin_w = cvRound( (double) hist_w/histSize);

		  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

		  /// Normalize the result to [ 0, histImage.rows ]
		  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

		  /// Draw for each channel
		  for( int i = 1; i < histSize; i++ )
		  {
		      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
		                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
		                       Scalar( 255, 0, 0), 2, 8, 0  );
		      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
		                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
		                       Scalar( 0, 255, 0), 2, 8, 0  );
		      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
		                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
		                       Scalar( 0, 0, 255), 2, 8, 0  );
		  }

		  /// Display
		  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
		  imshow("calcHist Demo", histImage );
		*/
