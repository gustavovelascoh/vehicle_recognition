/*
 * frame_saver.cpp
 *
 *  Created on: Jun 22, 2014
 *      Author: gustavo
 */

#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    VideoCapture cap;
    string input, name;

    if (argc < 3)
    {
        input = "../media_src/the_video.mp4";
        return -1;
    }
    else
    {
        input = argv[1];
        name = argv[2];
    }

    vector<int> frames_to_save;

    if (argc == 3)
    {
    	for(int i=1; i < 100; i++)
		{
			frames_to_save.push_back(i);
		}
    }
    else
    {
    	for(int i=3; i < argc; i++)
    	{
    		frames_to_save.push_back(atoi(argv[i]));
    	}
    }

    cout << "Frames to save:";

    for (int i=0;i< frames_to_save.size(); i++)
    {
    	cout << " " << frames_to_save[i] << ",";
    }

    cout << "\b. Total: " <<frames_to_save.size() << "." << endl;


	cap = VideoCapture(input);

	cout << "Opening: " << input << " -> ";
	if(!cap.isOpened())  // check if we succeeded
	{
		cout << "ERROR" << endl;
		return -1;
	}
	cout << "OK" << endl;

    for(;;)
    {
    	Mat frame;
        cap >> frame; // get a new frame from camera

        int curr_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
        std::stringstream ss;
		ss << name << curr_frame;
		std::string fns = ss.str();

		if(std::find(frames_to_save.begin(), frames_to_save.end(), curr_frame)!=frames_to_save.end())
		{
			std::cout << fns << std::endl;
			std::stringstream ln1, ln2, ln3, com;

			ln1 << "f" << curr_frame;
			ln1 << "_" << name << ".jpg";

			imwrite(ln1.str().c_str(),frame);
		}

        if (curr_frame >= frames_to_save[frames_to_save.size()-1])
		{
			return 0;
		}

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
