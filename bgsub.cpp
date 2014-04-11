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
	VideoCapture cap;
	
	for (int i=0; i < argc; i++)
        {
		if (!strcmp(argv[i], "-c"))
			use_cam = true;
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
	
	//Mat bg1m, bg1;
	//Mat out;
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
    << "USAGE: bgsub {-c | -f {filename} | -h}" << std::endl
    << "-c\tUse default camera" << std::endl
    << "-f\tUse file specified by filename" << std::endl
    << "-h\tDisplay this menu" << std::endl;
    //<< "" << endl;
    //<< "" << endl;
    
}