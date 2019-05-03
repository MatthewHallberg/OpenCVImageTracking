#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include "ARPipeline.hpp"
#include "DebugHelpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, const char * argv[]) {
        
    // Change this calibration to yours:
    CameraCalibration calibration(26.3219, 26.3219, 551.665, 661.338);
    
    // Read tracker image
    //string trackerFileName("card.jpg");
    //string trackerFileName("6ft.PNG");
    //string trackerFileName("dollar.jpg");
    string trackerFileName("pug.jpg");
    cout << "Reading tracker image : " << trackerFileName << endl;
    Mat patternImage = imread(trackerFileName, IMREAD_GRAYSCALE);
    
    ARPipeline pipeline(patternImage, calibration);
    
    //Capture stream from webcam.
    VideoCapture capture(0);
    
    //This variable will hold the image from the camera.
    Mat cameraFrame;
    //Read an image from the camera.
    capture.read(cameraFrame);
    
    //Optical flow stuff
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    vector<Point2f> detectionPoints;
    vector<Point2f> oldDetectionPoints;
    Mat previousCamFrame;
    
    int frameCount = 0;
    
    //Check if we can get the webcam stream.
    if(!capture.isOpened()) {
        cout << "Could not open camera" << std::endl;
        return -1;
    }
    
    while (true) {
        
        frameCount++;
        
        //Read an image from the camera.
        capture.read(cameraFrame);
        //convert to greyscale
        cvtColor(cameraFrame, cameraFrame, CV_BGR2GRAY);
        
        Size frameSize(cameraFrame.cols, cameraFrame.rows);
        
        //run detection
        if (frameCount % 20 == 0){
            detectionPoints.clear();
            //check if image is detected
            vector<Point2f> objectCorners = pipeline.processFrame(cameraFrame);
            if (objectCorners.size() > 0){
                
                //get 3d position
                PatternTrackingInfo trackingInfo = pipeline.getPatternLocation();
                //Transformation transformation = trackingInfo.pose3d;
                //Vector3 pos = transformation.t();
                //Matrix33 rot = transformation.r();
                
                //draw points on screen
                for (size_t i = 0; i < trackingInfo.points2d.size(); i++){
                  line(cameraFrame, trackingInfo.points2d[i], trackingInfo.points2d[ (i+1) % trackingInfo.points2d.size() ], Scalar(0,0,0), 2, cv::LINE_AA);
                }
                
                //get new match points
                detectionPoints = pipeline.m_patternDetector.getPoints();
                                
                //optical flow
//                cv::goodFeaturesToTrack(cameraFrame,// input, the image from which we want to know good features to track
//                                        detectionPoints,    // output, the points will be stored in this output vector
//                                        500,                  // max points, maximum number of good features to track
//                                        0.1,                // quality level, "minimal accepted quality of corners", the lower the more points we will get
//                                        .1,                  // minDistance, minimum distance between points
//                                        Mat(),               // mask
//                                        10,                   // block size
//                                        true,              // useHarrisDetector, makes tracking a bit better when set to true
//                                        0.001                 // free parameter for harris detector
//                                        );
                previousCamFrame = cameraFrame;
                oldDetectionPoints = detectionPoints;
            }
        } else if (!detectionPoints.empty()){
            
            vector<uchar> status;
            vector<float> err;
            vector<Point2f> flowDectionPoints;
            
            //not sure how this works yet shouldnt be image mask for first two params!!
            cv::calcOpticalFlowPyrLK(previousCamFrame,     // prev image
                                     cameraFrame,          // curr image
                                     oldDetectionPoints,           // find these points in the new image
                                     flowDectionPoints,           // result of found points
                                     status,               // output status vector, found points are set to 1
                                     err,                // each point gets an error value (see flag)
                                     cv::Size(40, 40),     // size of the window at each pyramid level
                                     7,                    // maxLevel - 0 = no pyramids, > 0 use this level of pyramids
                                     termcrit,             // termination criteria
                                     0.1,                    // flags OPTFLOW_USE_INITIAL_FLOW or OPTFLOW_LK_GET_MIN_EIGENVALS
                                     0.01                   // minEigThreshold
                                     );

            for (int i = 0; i < flowDectionPoints.size(); i++){
                Point2f center = Point2f((float)flowDectionPoints[i].x,(float)flowDectionPoints[i].y);
                circle(cameraFrame,center,10,Scalar(0,255,0),-1);
            }
            
            swap(oldDetectionPoints, flowDectionPoints);
            swap(previousCamFrame, cameraFrame);
        }
        
        //make window half the size
        resize(cameraFrame, cameraFrame, Size(cameraFrame.cols/2, cameraFrame.rows/2));
        namedWindow( "Camera", WINDOW_AUTOSIZE);
        imshow("Camera", cameraFrame);
        
        if (waitKey(50) >= 0)
            break;
    }
    return 0;
}
