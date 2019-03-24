#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "test1.hpp"
#include "test2.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, const char * argv[]) {
    //TestOne();
    //TestTwo();
    
    const int MAX_FEATURES = 300;//was 500
    const float MIN_MATCHES = 50;
    
    // Read tracker image
    string trackerFileName("card.jpg");
    //string trackerFileName("6ft.PNG");
    cout << "Reading tracker image : " << trackerFileName << endl;
    Mat trackerMat = imread(trackerFileName);
    
    //convert tracker to grayscale
    Mat trackerGray;
    cvtColor(trackerMat, trackerGray, COLOR_BGR2GRAY);

    // Detect ORB features and compute descriptors for tracker
    Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
    vector<KeyPoint> trackerKeyPoints;
    Mat trackerDescriptors;
    orb->detectAndCompute(trackerGray, Mat(), trackerKeyPoints, trackerDescriptors);
    
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
        
        //convert camera frame to grayscale
        Mat cameraGray;
        cvtColor(cameraFrame, cameraGray, COLOR_BGR2GRAY);
        
        // Detect ORB features and compute descriptors for camera
        Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
        Mat cameraDescriptors;
        vector<KeyPoint> cameraKeyPoints;
        orb->detectAndCompute(cameraGray, Mat(), cameraKeyPoints, cameraDescriptors);
    
        //cameraDescriptors.convertTo(cameraDescriptors, CV_32F);
    
        // Find 2 nearest matches
        vector<vector<DMatch>> matches;
        BFMatcher matcher;
        matcher.knnMatch(cameraDescriptors, trackerDescriptors, matches, 2);
        vector<cv::DMatch> good_matches;
        
        //Filter matches using the Lowes ratio test
        float ratio = .75;
        for (int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < ratio * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }
        
        cout << good_matches.size() << endl;
        //Draw matches
        Mat showMatches;
        drawMatches(cameraGray, cameraKeyPoints, trackerGray, trackerKeyPoints, good_matches, showMatches);
        //show camera frame in window called output
        imshow("output", showMatches);

        //Waits 50 miliseconds for key press, returns -1 if no key is pressed during that time
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
    

}
