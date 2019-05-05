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
    
    int detectionFrame = 15;
    
    //Optical flow stuff
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,40,0.0001);
    vector<Point2f> detectionPoints;
    vector<Point2f> oldDetectionPoints;
    Mat previousCamFrame;
    
    //detection keypoints and extractors
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create(100);
    std::vector<cv::KeyPoint> detectedKeypoints;
    Mat detectedDescriptors;
    
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
        if (frameCount % detectionFrame == 0){
            detectionPoints.clear();
            //check if image is detected
            vector<Point2f> objectCorners = pipeline.processFrame(cameraFrame);
            if (objectCorners.size() > 0){
                //get 3d position
                //PatternTrackingInfo trackingInfo = pipeline.getPatternLocation();
                //Transformation transformation = trackingInfo.pose3d;
                //Vector3 pos = transformation.t();
                //Matrix33 rot = transformation.r();
                
                //draw points on screen (angled box)
               // for (size_t i = 0; i < objectCorners.size(); i++){
                 // line(cameraFrame, objectCorners[i], objectCorners[ (i+1) % objectCorners.size() ], Scalar(0,0,0), 2, cv::LINE_AA);
                //}
                
                //get points to track from detection
                //detectionPoints = pipeline.m_patternDetector.getPoints();
                
                //first draw box instead of polygon
                int width = objectCorners[1].x - objectCorners[0].x;
                int height = objectCorners[2].y - objectCorners[0].y;
                Rect box(objectCorners[0].x,objectCorners[0].y,width,height);
                
                //check if box is in bounds of camera
                if ((box & cv::Rect(0, 0, cameraFrame.cols, cameraFrame.rows)) == box){
                    
                    //draw box around detection
                    rectangle(cameraFrame, box, Scalar(0,0,0), 2, 1 );
                    
                    //get keypoints from detection box
                    Mat boxImage = cameraFrame(box);
                    extractor->detectAndCompute(boxImage, noArray(), detectedKeypoints, detectedDescriptors);
                    
                    //get features we will track for optical flow
                    cv::goodFeaturesToTrack(cameraFrame,// input, the image from which we want to know good features to track
                                            detectionPoints,    // output, the points will be stored in this output vector
                                            500,                  // max points, maximum number of good features to track
                                            0.1,                // quality level, "minimal accepted quality of corners", the lower the more points we will get
                                            .01,                  // minDistance, minimum distance between points
                                            Mat(),               // mask
                                            5,                   // block size
                                            false,              // useHarrisDetector, makes tracking a bit better when set to true
                                            0.01                 // free parameter for harris detector
                                            );
                    
                    vector<Point2f> goodPoints;
                    //remove features not in box
                    for( int i = 0; i < detectionPoints.size(); i++ ) {
                        if (detectionPoints[i].x > objectCorners[0].x && detectionPoints[i].x < objectCorners[0].x + width &&
                            detectionPoints[i].y > objectCorners[0].y && detectionPoints[i].y < objectCorners[0].y + height){
                            goodPoints.push_back(detectionPoints[i]);
                        }
                    }
                    
                    if (goodPoints.size() > 3){
                        previousCamFrame = cameraFrame;
                        oldDetectionPoints = goodPoints;
                        detectionFrame = 5000;
                    } else {
                        detectionPoints.clear();
                    }
                }
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
                                     cv::Size(80, 80),     // size of the window at each pyramid level
                                     4,                    // maxLevel - 0 = no pyramids, > 0 use this level of pyramids
                                     termcrit,             // termination criteria
                                     0.1,                    // flags OPTFLOW_USE_INITIAL_FLOW or OPTFLOW_LK_GET_MIN_EIGENVALS
                                     0.01                   // minEigThreshold
                                     );
            
            //make box around bounds of flow tracking points
            Rect bounds = boundingRect(flowDectionPoints);
            
            //get keypoints from detection box if in bounds
            if ((bounds & cv::Rect(0, 0, cameraFrame.cols, cameraFrame.rows)) == bounds){
                Mat boxImage = cameraFrame(bounds);
                
                vector<KeyPoint> trackedKeypoints;
                Mat trackedDescriptors;
                extractor->detectAndCompute(boxImage, noArray(), trackedKeypoints, trackedDescriptors);
                
                if (trackedKeypoints.size() > detectedKeypoints.size()/3){
                    
                    //find matches
                    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
                    std::vector< std::vector<DMatch> > knn_matches;
                    matcher->knnMatch( trackedDescriptors, detectedDescriptors, knn_matches, 2 );
                    //-- Filter matches using the Lowe's ratio test
                    const float ratio_thresh = 0.7f;
                    vector<DMatch> good_matches;
                    for (size_t i = 0; i < knn_matches.size(); i++) {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
                            good_matches.push_back(knn_matches[i][0]);
                        }
                    }
                    
                    if (good_matches.size() > 0){
                        //find all points that match and convert from local space to screen space
                        vector<Point2f> matchPoints;
                        for (int i = 0; i < good_matches.size(); i++){
                            Point2f point = trackedKeypoints[good_matches[i].queryIdx].pt;
                            point.x += bounds.x;
                            point.y += bounds.y;
                            matchPoints.push_back(point);
                        }
                        
                        //draw all points
                        for (int i = 0; i < matchPoints.size(); i++){
                            circle(cameraFrame,matchPoints[i],10,Scalar(0,255,0),-1);
                        }
                        
                        //draw box from optical flow points
                        Rect matchBounds = boundingRect(matchPoints);
                        rectangle(cameraFrame, matchBounds, Scalar(0,0,0), 2, 1);
                        
                        swap(oldDetectionPoints, flowDectionPoints);
                        swap(previousCamFrame, cameraFrame);
                    } else {
                        //stop tracking and start detecting again
                        detectionFrame = 5;
                        detectionPoints.clear();
                    }
                } else {
                    //stop tracking and start detecting again
                    detectionFrame = 5;
                    detectionPoints.clear();
                }
            } else {
                //stop tracking and start detecting again
                detectionFrame = 5;
                detectionPoints.clear();
            }
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
