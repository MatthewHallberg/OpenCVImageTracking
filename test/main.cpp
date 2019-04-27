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
    Size subPixWinSize(10,10), winSize(31,31);
    vector<Point2f> corners;
    vector<Point2f> oldCorners;
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
        if (frameCount % 10 == 0){
            corners.clear();
            //check if image is detected
            vector<Point2f> objectPoints = pipeline.processFrame(cameraFrame);
            if (objectPoints.size() > 0){
                
                //get 3d position
                //Transformation transformation = pipeline.getPatternLocation();
                //Vector3 pos = transformation.t();
                //Matrix33 rot = transformation.r();
                
                //draw points on screen
                //for (size_t i = 0; i < objectPoints.size(); i++){
                //  line(cameraFrame, objectPoints[i], objectPoints[ (i+1) % objectPoints.size() ], Scalar(255,0,0), 2, cv::LINE_AA);
                //}
            
                //find box to track
                int width = objectPoints[1].x - objectPoints[0].x;
                int height = objectPoints[2].y - objectPoints[0].y;
                Rect2d box(objectPoints[0].x,objectPoints[0].y,width,height);
                //Point2f center = Point2f((float)objectPoints[0].x + width/2,(float)objectPoints[0].y + height/2);
                
                //draw rectangle around detection
                rectangle(cameraFrame, box, Scalar(0,255,0), 2, 1 );
                
                //draw circle at center
                //circle(cameraFrame,center,50,Scalar(255,0,0),-1);
                
                //make sure box is inside plane i.e detection is good
                if (0 <= box.x && 0 <= box.width && box.x + box.width <= cameraFrame.cols
                    && 0 <= box.y && 0 <= box.height && box.y + box.height <= cameraFrame.rows){
                    corners.clear();
                    //optical flow
                    cv::goodFeaturesToTrack(cameraFrame,            // input, the image from which we want to know good features to track
                                            corners,    // output, the points will be stored in this output vector
                                            100,                  // max points, maximum number of good features to track
                                            0.1,                // quality level, "minimal accepted quality of corners", the lower the more points we will get
                                            1,                  // minDistance, minimum distance between points
                                            Mat(),               // mask
                                            4,                   // block size
                                            true,              // useHarrisDetector, makes tracking a bit better when set to true
                                            0.04                 // free parameter for harris detector
                                            );
                    cornerSubPix(cameraFrame, corners, subPixWinSize, Size(-1,-1), termcrit);
                    previousCamFrame = cameraFrame;
                    oldCorners = corners;
                }
            }
        } else if (!corners.empty()){
            
            vector<uchar> status;
            vector<float> err;
            vector<Point2f> newCorners;
            
            //not sure how this works yet shouldnt be image mask for first two params!!
            cv::calcOpticalFlowPyrLK(previousCamFrame,         // prev image
                                     cameraFrame,          // curr image
                                     oldCorners,           // find these points in the new image
                                     newCorners,           // result of found points
                                     status,               // output status vector, found points are set to 1
                                     err,                // each point gets an error value (see flag)
                                     cv::Size(31, 31),     // size of the window at each pyramid level
                                     3,                    // maxLevel - 0 = no pyramids, > 0 use this level of pyramids
                                     termcrit,             // termination criteria
                                     0,                    // flags OPTFLOW_USE_INITIAL_FLOW or OPTFLOW_LK_GET_MIN_EIGENVALS
                                     0.1                   // minEigThreshold
                                     );
            
            Point2f center = Point2f((float)newCorners[0].x,(float)newCorners[0].y);
            circle(cameraFrame,center,50,Scalar(255,0,0),-1);

            
            //show new point
            int width = newCorners[1].x - newCorners[0].x;
            int height = newCorners[2].y - newCorners[0].y;
            Rect2d box(newCorners[0].x,newCorners[0].y,width,height);
            
            //draw rectangle around detection
            rectangle(cameraFrame, box, Scalar(0,255,0), 2, 1 );
            
            swap(oldCorners, newCorners);
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
