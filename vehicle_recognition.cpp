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
        
        
        //bgsub1.getBackgroundImage(bg1);
        
        imshow("Origin", frame);
        imshow("Background mask", bg_mask);
        imshow("Opening", frame_masked);
        
        key = waitKey(30);
        
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

void print_help()
{
    std::cout
    << "bgsub: Background substraction" << std::endl
    << "" << std::endl
    << "USAGE: bgsub {-c | -f {filename} | -h} [{-mog [history nog br]| -mog2 [history thr shdet]}]" << std::endl
    << "-c\tUse default camera" << std::endl
    << "-f\tUse file specified by filename" << std::endl
    << "-h\tDisplay this menu" << std::endl
    << "" << std::endl
    << "-mog\tUse BackgroundSubtractorMOG class [default]" << std::endl
    << "\thistory\tLength of the history. [default = 500]" << std::endl
    << "\tnog\tNumber of Gaussian mixtures. [default = 5]" << std::endl
    << "\tbr\tBackground ratio. [default = 0.5]" << std::endl
    << "" << std::endl
    << "-mog2\tUse BackgroundSubtractorMOG2 class:" << std::endl
    << "\thistory\tLength of the history. [default = 500]" << std::endl
    << "\tthr\tThreshold. [default = 16]" << std::endl
    << "\tshdet\tEnable/disable shadow detection [default = True]" << std::endl
    << "" << std::endl
    << "If neither -mog nor mog2 optiones are selected, it will take default values." << std::endl;
    //<< "" << std::endl
    //<< "" << std::endl
    //<< "" << std::endl
    
}