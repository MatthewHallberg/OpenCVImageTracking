#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "test1.hpp"
#include "test2.hpp"

using namespace std;
using namespace cv;

int TestOne(){
    
    //Capture stream from webcam.
    VideoCapture capture(0);
    
    //Check if we can get the webcam stream.
    if(!capture.isOpened()) {
        cout << "Could not open camera" << std::endl;
        return -1;
    }
    
    while (true) {
        //This variable will hold the image from the camera.
        Mat cameraFrame;
        
        //Read an image from the camera.
        capture.read(cameraFrame);
        
        //show camera frame in window called output
        imshow("output", cameraFrame);
        
        //Waits 50 miliseconds for key press, returns -1 if no key is pressed during that time
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
}
