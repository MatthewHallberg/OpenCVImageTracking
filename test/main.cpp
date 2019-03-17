//
//  main.cpp
//  test
//
//  Created by matthew hallberg on 3/17/19.
//  Copyright © 2019 matthew hallberg. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, const char * argv[]) {

    //Capture stream from webcam.
    cv::VideoCapture capture(0);
    
    //Check if we can get the webcam stream.
    if(!capture.isOpened())
    {  
        std::cout << "Could not open camera" << std::endl;
        return -1;
    }
    
    while (true)
    {
        //This variable will hold the image from the camera.
        cv::Mat cameraFrame;
        
        //Read an image from the camera.
        capture.read(cameraFrame);
        
        //show cam background
        cv::imshow("output", cameraFrame);
        
        //Waits 50 miliseconds for key press, returns -1 if no key is pressed during that time
        if (cv::waitKey(50) >= 0)
            break;
    }
    
    return 0;
    
}
