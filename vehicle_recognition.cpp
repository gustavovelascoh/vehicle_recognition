#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void print_help();

int main(int argc, char** argv)
{
    int key = 0;
    VideoCapture cap;
    std::string input;
    std::vector<int> frames_to_save;
    if (argc < 2)
    {
        input = "../media_src/ND.mp4";
    }
    else
    {
        input = argv[1];
    }
    
    cap = VideoCapture(input);
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Opening file " << input << std::endl;
    std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
    
    if(!cap.isOpened())  // check if we succeeded
	return -1;
    
    BackgroundSubtractorMOG2 bg_sub(700,32,false);
    
    Mat element = getStructuringElement(MORPH_RECT, Size(7,7),Point(3,3));
    namedWindow("Origin",1);
    namedWindow("Background mask",1);
    namedWindow("Opening",1);
    for(;;)
    {
        Mat frame, bg_mask, frame_masked, op_mask;
        cap >> frame; // get a new frame from camera
        
        //cvtColor(frame, frame, CV_BGR2GRAY);
        GaussianBlur(frame, frame, Size(5,5), 1.5, 1.5);
        
        bg_sub(frame,bg_mask);
        
        morphologyEx(bg_mask, op_mask, CV_MOP_CLOSE, element);
        
        frame.copyTo(frame_masked,op_mask);
        
        std::stringstream ss;
        rectangle(frame,cv::Point(150,5),cv::Point(250,25),cv::Scalar(255,255,255,-1));
        ss << "frame " << cap.get(CV_CAP_PROP_POS_FRAMES);
        std::string fns = ss.str();

        putText(frame,fns.c_str(),cv::Point(165,20),FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
        //bgsub1.getBackgroundImage(bg1);
        
        imshow("Origin", frame);
        imshow("Background mask", bg_mask);
        imshow("Opening", frame_masked);
        
        key = waitKey(15);
        
        if (key == 112)
        {
            while(key < 0 || key == 112)
            {
                key = waitKey(30);
            }
        }        
        else if(key >= 0)
        {
            break;
        }
        //if(waitKey(30) == )
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}