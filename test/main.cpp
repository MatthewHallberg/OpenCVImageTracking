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
    
    const int MAX_FEATURES = 1000;//was 500
    const int MIN_FEATURES = 15;
    
    // Read tracker image
    //string trackerFileName("card.jpg");
    //string trackerFileName("6ft.PNG");
    string trackerFileName("dollar.jpg");
    cout << "Reading tracker image : " << trackerFileName << endl;
    Mat trackerGray = imread(trackerFileName, IMREAD_GRAYSCALE);

    //Create detector and matcher
    Ptr<ORB> detector = ORB::create(MAX_FEATURES);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    
    //Detect ORB features and compute descriptors for tracker
    vector<KeyPoint> trackerKeyPoints;
    Mat trackerDescriptors;
    detector->detectAndCompute(trackerGray, Mat(), trackerKeyPoints, trackerDescriptors);
    
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
        
        // Detect ORB features and compute descriptors for camera
        Mat cameraDescriptors;
        vector<KeyPoint> cameraKeyPoints;
        detector->detectAndCompute(cameraFrame, Mat(), cameraKeyPoints, cameraDescriptors);
        
        // Find 2 nearest matches
        vector<vector<DMatch> > knn_matches;
        matcher->knnMatch( cameraDescriptors, trackerDescriptors, knn_matches, 2 );
        
        //Filter matches using the Lowes ratio test
        const float ratio_thresh = 0.75f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++){
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        
        cout << good_matches.size() << endl;
        
        //dont draw matches if less than threshold
        if (good_matches.size() < MIN_FEATURES){
            good_matches.clear();
        }
        
        Mat showMatches;
        drawMatches(cameraFrame, cameraKeyPoints, trackerGray, trackerKeyPoints, good_matches, showMatches);
        imshow("output", showMatches);
        
        //Waits 50 miliseconds for key press, returns -1 if no key is pressed during that time
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
}
