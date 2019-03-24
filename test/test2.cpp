#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "test1.hpp"
#include "test2.hpp"

using namespace std;
using namespace cv;

int TestTwo(){
    
    // Read tracker image
    string trackerFileName("card.jpg");
    cout << "Reading tracker image : " << trackerFileName << endl;
    Mat trackerMat = imread(trackerFileName);
    
    // Read image to be aligned
    string sceneFilename("scene.jpg");
    cout << "Reading image to align : " << sceneFilename << endl;
    Mat sceneMat = imread(sceneFilename);
    
    
    
    
    return 0;
}
