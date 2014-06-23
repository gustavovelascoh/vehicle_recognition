#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"

using namespace cv;

void print_help();

int main(int argc, char** argv)
{
	//VideoCapture cap(0); // open the default camera
	//http://webcam.prvgld.nl/n224.html
	//VideoCapture cap("http://cam1.infolink.ru/mjpg/video.mjpg");
	//VideoCapture cap("http://popa.datacom.bg/mjpg/video.mjpg");
	//http://194.218.96.93/axis-cgi/mjpg/video.cgi
	//http://166.141.19.4/mjpg/video.mjpg
        print_help();
        if (argc == 1 )
	{
            return -1;
        }
        
	bool use_cam = false;
        int use_mog = 1;
        int history = 500;
        int nog = 5;
        float bgr = 0.5;
        
	VideoCapture cap;
	
	for (int i=0; i < argc; i++)
        {
            if (!strcmp(argv[i], "-c"))
		use_cam = true;
            if (!strcmp(argv[i], "-mog"))
                use_mog = 1;
            if (!strcmp(argv[i], "-mog2"))
                use_mog = 2;
            if (!strcmp(argv[i], "history"))
                use_mog = 2;
	}
	if (use_cam)
	{
		cap = VideoCapture(0);
                std::cout << "Opening camera" << std::endl;
	}
	else
	{	
		std::string input = argv[1];
		cap = VideoCapture(input);
		std::cout << "Opening custom source: " << input << std::endl;		
	}	
	
	if(!cap.isOpened())  // check if we succeeded
		return -1;
	
        if (use_mog != 0 && (argc < 6 && argc > (3 + (use_cam)?0:1)))
        {
            std::cout << "Missing arguments: Receive " << argc << " arguments, expected 6 or 7" << std::endl;
            return -1;
        }
            
	//Mat bg1m, bg1;
	//Mat out;
        
        /* BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0)
        /* Parameters:	
        /* history – Length of the history.
        /* nmixtures – Number of Gaussian mixtures.
        /* backgroundRatio – Background ratio.
        /* noiseSigma – Noise strength.
        /* 
        */
        BackgroundSubtractor bg_sub;
        
	BackgroundSubtractorMOG bgsub1(500,5,0.5);
	BackgroundSubtractorMOG2 bgsub2(500,16,false);
	namedWindow("CamCap",1);
	namedWindow("bg1",1);
	namedWindow("bg2",1);
	for(;;)
	{
		Mat frame;
		Mat bg1m, bg2m;
		Mat out1, out2;
		cap >> frame; // get a new frame from camera
		
		//cvtColor(frame, frame, CV_BGR2GRAY);
		//GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
		
		bgsub1(frame,bg1m);
		bgsub2(frame,bg2m);
		frame.copyTo(out1,bg1m);
		frame.copyTo(out2,bg2m);
		
		//bgsub1.getBackgroundImage(bg1);
		
		imshow("CamCap", frame);
		imshow("bg1", out1);
		imshow("bg2", out2);
		
		if(waitKey(30) >= 0) break;
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