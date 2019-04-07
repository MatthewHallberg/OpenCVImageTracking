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
    vector<Rect> boxes;
    
    // Create multitracker
    Ptr<MultiTracker> multiTracker = MultiTracker::create();
    
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
        if (frameCount % 5 == 0){
            boxes.clear();
            multiTracker->clear();
            multiTracker = MultiTracker::create();
            cout << "detecting: " << frameCount << endl;
            //multiTracker.reset();
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
                if (0 <= box.x && 0 <= box.width && box.x + box.width <= cameraFrame.cols
                    && 0 <= box.y && 0 <= box.height && box.y + box.height <= cameraFrame.rows){
                    // box within the image plane
                    boxes.push_back(box);
                    for(int i=0; i < boxes.size(); i++){
                        //use CSRT, KCF, or MOSSE maybe MedianFlow
                        multiTracker->add(TrackerMedianFlow::create(), cameraFrame, Rect2d(boxes[i]));
                    }
                }
            }
        } else {
            cout << "tracking: " << frameCount << endl;
            //run tracking
            multiTracker->update(cameraFrame);
            // Draw tracked objects
            for(unsigned i=0; i<multiTracker->getObjects().size(); i++){
                rectangle(cameraFrame, multiTracker->getObjects()[i], Scalar(255,0,0), 2, 1);
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
