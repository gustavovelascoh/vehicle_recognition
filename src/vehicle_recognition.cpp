#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Bgsub.h"
#include <cmath>

using namespace cv;
using namespace std;

RNG rng(12345);

void show_cap_info(VideoCapture cap, std::string);
void morph_ops(InputArray src, OutputArray dst);
int find_similar_contour(vector<Point> cont, vector<vector<Point> > cont_vec);

int main(int argc, char** argv)
{
    int key = 0, crop = 0;
    int save_frames = 0;
    int gui = 1;
    int save_video = 0, steps = 0;
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
    	if (!strcmp(argv[1],"-s"))
    	{
    		steps = 1;
    		input = "../media_src/the_video.mp4";
    		crop = 1;
    	}
    	else
    	{
    		input = argv[1];
    	}
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
    Mat bg_mask_1, thr_1, track_image;
    //Vector of contours
    vector<vector<Point> > objects;
    vector<vector<Point> > paths;

    int obj_ind=0, path_ind=0;



    for(;;)
    {
    	/* Get current frame number */
    	int curr_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
		std::stringstream ss;
		ss << "frame " << curr_frame;
		std::string fns = ss.str();

		// start timer
    	tp = (double)getTickCount();

        Mat frame,  bg_mask, bg_mask_ns, frame_masked, mask_morph, mask_rgb;//, cl_mask, op_mask;
        cap >> frame; // get a new frame from camera
        
        if (crop)
        	frame = frame.colRange(160,1120);

        //Size Sf = frame.cols;
        //std::cout << frame.cols << "x" << frame.rows << std::endl;

        //cvtColor(frame, frame, CV_BGR2YCrCb);
        //cvtColor(frame, frame, CV_BGR2GRAY);


        GaussianBlur(frame, frame, Size(5,5), 1.5, 1.5);
        
        // Execute background subtraction and get mask
        bg_sub(frame,bg_mask);

        cout << "SUM: " << sum(bg_mask)[0] << endl;

        if (sum(bg_mask)[0] < 1)
        	bg_mask_1.copyTo(bg_mask);

        // Apply thresholding to omit shadow
        threshold(bg_mask,bg_mask_ns,200,255,0);


        //if (curr_frame > 1)
        //	bitwise_or(bg_mask_ns, bg_mask_1, bg_mask_ns);

        morph_ops(bg_mask_ns,mask_morph);

        // mask original frame
        frame.copyTo(frame_masked,mask_morph);
        
        //if (curr_frame > 1)
        bg_mask_ns.copyTo(bg_mask_1);


        // valid contours
        vector<vector<Point> > contours;
        // all_contours (raw)
		vector<vector<Point> > contours_r;
		Mat frame_gray, threshold_output;
		vector<Vec4i> hierarchy;

//		cvtColor( mask_morph, frame_gray, CV_BGR2GRAY );
		mask_morph.copyTo(frame_gray);

		/// Detect edges using Threshold
		threshold( frame_gray, threshold_output, 64, 255, THRESH_BINARY );

		//threshold_output.copyTo(bg_mask_ns);
		/// Find contours
		findContours( threshold_output, contours_r, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		cout << "contours #: " << contours_r.size() << endl;

		if ((contours_r.size() == 0) && ( curr_frame > 1))
		{
			findContours( thr_1, contours_r, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		}
		else
		{
			threshold_output.copyTo(thr_1);
		}

		for( int i = 0; i < contours_r.size(); i++ )
		{
			if ((arcLength(contours_r[i],true) > 180) && (contourArea(contours_r[i],false) > 400))
			{
				contours.push_back(contours_r[i]);

				Moments c_mom = moments(contours_r[i]);
				double c_x = c_mom.m10/c_mom.m00;
				double c_y = c_mom.m01/c_mom.m00;
				Point c_point = Point(c_x,c_y);
				vector<Point> path;
				if (objects.size() == 0)
				{
					cout << "size 0" << endl;
					objects.push_back(contours_r[i]);
					path.push_back(c_point);
					paths.push_back(path);
				}
				else
				{
					cout << "size 1" << endl;
					int s_ind = find_similar_contour(contours_r[i], objects);

					if (s_ind >= 0)
					{
						objects[s_ind] = contours_r[i];
						paths[s_ind].push_back(c_point);

						cout << "path " << paths[s_ind] << endl;
					}
					else
					{
						objects.push_back(contours_r[i]);
						path.push_back(c_point);
						paths.push_back(path);
					}
				}



			}

		}

		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );

		for( int i = 0; i < contours.size(); i++ )
		{
			cout << "Contour " << i << " size: " << contours[i].size() << ". ";
								cout << "arc_length: " << arcLength(contours[i],true) << ". ";
								cout << "area: " << contourArea(contours[i],false) << ". ";
								cout << "(" << contours[i][0].x << "," << contours[i][0].y << ")" << endl;

			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		//minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
		}
		cout << "cont size: " << contours.size() << endl;
		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

		frame.copyTo(drawing);
		for( int i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
		}

		for( int i = 0; i< paths.size(); i++ )
		{
			drawContours( drawing, paths, i, Scalar(0,0,0), 1, 8, vector<Vec4i>(), 0, Point() );
		}

        // Execution time calculations
        tp = ((double)getTickCount() - tp)/getTickFrequency();
        ex_ts.push_back(tp);

        if (tp > t_max)
        	t_max = tp;

        if (tp < t_min)
        	t_min = tp;

        t_avg += tp;


		



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
			imshow("Frame", bg_mask);
			imshow("Mask", drawing);


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

        if (steps)
        	waitKey(0);

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
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(2,2));
	Mat temp;

	morphologyEx(src, temp, CV_MOP_OPEN, element, Point(1,1),3);
	morphologyEx(temp, dst, CV_MOP_CLOSE, element2, Point(1,1), 5);
}

int find_similar_contour(vector<Point> cont, vector<vector<Point> > cont_vec)
{
	double c_len = arcLength(cont,true);
	double c_area = contourArea(cont,false);
	int ind = 0;

	Moments c_mom = moments(cont);
	double c_x = c_mom.m10/c_mom.m00;
	double c_y = c_mom.m01/c_mom.m00;

	cout << "MOMS: " << c_mom.m00 << ", " << c_mom.m01 << ", " << c_mom.m10 << " -> " << c_x << ". " << c_y << endl;

	double olen, oarea;

	vector<double> d_len_v , d_area_v, d_cm_v;
	double d_len, d_area, d_cm;

	for (vector<vector<Point> >::iterator it = cont_vec.begin() ; it != cont_vec.end(); ++it)
	{
		d_len = abs(arcLength(*it,true)-c_len);
		d_area = abs(contourArea(*it,false)-c_area);

		Moments d_mom = moments(*it);
		double d_x = abs((d_mom.m10/d_mom.m00) - c_x);
		double d_y = abs((d_mom.m01/d_mom.m00) - c_y);

		cout << "deltas [" << ind << "]: (" << d_len << ", " << d_area << ")-(" << d_x << ", " << d_y << ")";

		if ((d_x < 16) && (d_y < 16))
		{
			cout << " = MATCH" << endl;
			return ind;
		}
		else
			cout << endl;

		ind++;
	}

	return -1;
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
