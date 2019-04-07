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
    
    int frameCount = 0;
    Rect2d box(287, 23, 86, 320);
    
    //create tracker
    Ptr<Tracker> tracker = TrackerBoosting::create();
    tracker->init(cameraFrame, box);
    
    //Check if we can get the webcam stream.
    if(!capture.isOpened()) {
        cout << "Could not open camera" << std::endl;
        return -1;
    }
    
    while (true) {
        
        frameCount++;
        
        //Read an image from the camera.
        capture.read(cameraFrame);
        
        Size frameSize(cameraFrame.cols, cameraFrame.rows);
        
        //run detection
        if (frameCount % 15 == 0){
            //check if image is detected
            std::vector<cv::Point2f> objectPoints = pipeline.processFrame(cameraFrame);
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
                rectangle(cameraFrame, box, Scalar(255,0,0), 2, 1 );
            } else {
                //run tracking
                // Update the tracking result
                bool ok = tracker->update(cameraFrame, box);
                if (ok){
                    // Tracking success : Draw the tracked object
                    rectangle(cameraFrame, box, Scalar( 255, 0, 0 ), 2, 1 );
                } else {
                    // Tracking failure detected.
                    putText(cameraFrame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
                }
                // Display tracker type on frame
                putText(cameraFrame, "Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
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
