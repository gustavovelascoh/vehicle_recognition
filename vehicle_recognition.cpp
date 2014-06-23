#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Bgsub.h"

using namespace cv;
void show_cap_info(VideoCapture cap, std::string);
void morph_ops(InputArray src, OutputArray dst);

int main(int argc, char** argv)
{
    int key = 0, crop = 0;
    int save_frames = 0;
    int gui = 0;
    int save_video = 1;
    const string NAME = "mask_morph.avi";   // Form the new name with container

    VideoCapture cap;
    std::string input;

    if (argc < 2)
    {
        input = "../media_src/the_video.mp4";
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

    std::cout << "OpenCV ver: " << CV_VERSION <<  std::endl;
    /* ---------- Frame saving ---------- */

    //static const int arr[] = {2160, 2260, 4380, 6170, 6390};
    static const int arr[] = {100, 200, 300, 400, 500};
    //static const int arr[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000};
    std::vector<int> frames_to_save (arr, arr + sizeof(arr) / sizeof(arr[0]) );
    

    VideoWriter outputVideo;
    /* ---------- Video save ---------- */
    if (save_video)
    {
		Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
						  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		//const string NAME = "mask_morph.avi";   // Form the new name with container
		int ex;
		//VideoWriter outputVideo;

		//ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
		//ex = CV_FOURCC('M','J','P','G');
		//ex = CV_FOURCC('P','I','M','1');
		//ex = CV_FOURCC('A','E','M','I');
		//ex = CV_FOURCC('X','2','6','4');
		//ex = CV_FOURCC('D','A','V','C');
		//ex = CV_FOURCC('F','M','P','4');
		//ex = CV_FOURCC('D', 'I', 'V', '3');
		//ex = CV_FOURCC('M', 'P', '4', '2');
		//ex = CV_FOURCC('D', 'I', 'V', 'X');
		//ex = CV_FOURCC('I', '2', '6', '3');// OpenCV Error: Unsupported format or combination of formats
		ex = CV_FOURCC('M', 'P', 'E', 'G');
		//ex = CV_FOURCC('D', 'I', 'V', '3');

		char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};

			std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
					 << " of nr#: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
			std::cout << "Input codec type: " << EXT << std::endl;

		//outputVideo.open(NAME, ex, cap.get(CV_CAP_PROP_FPS), Size(960,720), true);
		outputVideo.open(NAME, ex, cap.get(CV_CAP_PROP_FPS), Size(960,720), false);
		//outputVideo.open(NAME, -1, cap.get(CV_CAP_PROP_FPS), S, false);

		//outputVideo.open(NAME, CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), S, true);
		//outputVideo.open(NAME, CV_FOURCC('P','I','M','1'), cap.get(CV_CAP_PROP_FPS), S, true);
		//outputVideo.open(NAME, CV_FOURCC('A','E','M','I'), cap.get(CV_CAP_PROP_FPS), S, true);

		if (outputVideo.isOpened())
		{
			std::cout << "Opened succesfully" << std::endl;
		}
		else
		{
			std::cout << "NOT Opened succesfully" << std::endl;
		}
    }

    /*


    /* ---------- Background extractor ---------- */
    int history =  400;
    float varThreshold = 12;
    
    //BackgroundSubtractorMOG2 bg_sub(history,varThreshold,true);
    Bg_sub bg_sub(history,varThreshold,true,0.5);
    
    //bg_sub.fTau = 0.1;

    std::cout << "fTau = " << bg_sub.get_fTau() << std::endl;
    std::cout << "history = " << bg_sub.get_history() << std::endl;
    //bg_sub.BackgroundSubtractorMOG2::fTau = 0.3;
    //,BackgroundSubtractorMOG2::fTau = 0.3

    /* ----------  ---------- */
    std::stringstream ln1, ln2, ln3;
    //Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(2,2));
    if (gui)
    {
    	namedWindow("Frame",1);
    	namedWindow("Mask",1);
    }
    //namedWindow("Closing(Frame & Mask)",1);
    double t = (double)getTickCount();
    double tp, t_max=0.0, t_min = 10.0, t_avg=0.0;
    Vector<double_t> ex_ts;
    //return 1;
    for(;;)
    {
    	tp = (double)getTickCount();
        Mat frame,  bg_mask, bg_mask_ns, frame_masked, mask_morph, mask_rgb;//, cl_mask, op_mask;
        cap >> frame; // get a new frame from camera
        
        if (crop)
        	frame = frame.colRange(160,1120);

        //Size Sf = frame.cols;
        //std::cout << frame.cols << "x" << frame.rows << std::endl;

        //cvtColor(frame, frame, CV_BGR2YCrCb);
        cvtColor(frame, frame, CV_BGR2GRAY);


        GaussianBlur(frame, frame, Size(5,5), 1.5, 1.5);
        
        // Execute background subtraction and get mask
        bg_sub(frame,bg_mask);
        // Apply thresholding to omit shadow
        threshold(bg_mask,bg_mask_ns,200,255,0);

        morph_ops(bg_mask_ns,mask_morph);

        // mask original frame
        frame.copyTo(frame_masked,mask_morph);
        
        tp = ((double)getTickCount() - tp)/getTickFrequency();
        ex_ts.push_back(tp);

        if (tp > t_max)
        	t_max = tp;

        if (tp < t_min)
        	t_min = tp;

        t_avg += tp;

        int curr_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
        std::stringstream ss;
		ss << "frame " << curr_frame;
		std::string fns = ss.str();
		



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
				ln1 << "sh05v12.jpg";

				//ln1 << ".jpg";
				//ln2 << "_b" << com.str();
				//ln3 << "_m" << com.str();

				//imwrite(ln1.str().c_str(),frame_masked);
				imwrite(ln1.str().c_str(),bg_mask);
				//imwrite(ln3.str().c_str(),frame_masked);
			}
        }else
        {
        	/*
        	if (curr_frame % 10)
        		std::cout << fns << '\r';
        	*/
        	std::cout << fns << ": " << tp << "s" << std::endl;
        }

        // save video
        if (save_video)
        {
        	//cvtColor(mask_morph,mask_rgb,CV_GRAY2RGB);
        	outputVideo.write(mask_morph);
        }
        //outputVideo << bg_mask;

        // ------ Show images ------
        if (gui)
        {
			imshow("Frame", frame);
			imshow("Mask", frame_masked);


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

        if (curr_frame >= cap.get(CV_CAP_PROP_FRAME_COUNT))
		{
			t = ((double)getTickCount() - t)/getTickFrequency();
			std::cout << std::endl;
			std::cout << "Time elapsed: " << t << " secs" << std::endl;
			std::cout << "Sum of tp: " << t_avg << ". " << t_min << "/" << t_avg/curr_frame << "/" << t_max << std::endl;
			return 0;
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
	        std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height << std::endl;
	        std::cout << "#frames: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << ", fps: " << cap.get(CV_CAP_PROP_FPS) << std::endl;
}

void morph_ops(InputArray src, OutputArray dst)
{
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3,3),Point(1,1));
	Mat temp;

	morphologyEx(src, temp, CV_MOP_OPEN, element, Point(1,1),3);
	morphologyEx(temp, dst, CV_MOP_CLOSE, element, Point(1,1), 4);
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
