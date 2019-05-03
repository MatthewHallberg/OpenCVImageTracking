#ifndef ARPIPELINE_HPP
#define ARPIPELINE_HPP

////////////////////////////////////////////////////////////////////
// File includes:
#include "PatternDetector.hpp"
#include "CameraCalibration.hpp"
#include "GeometryTypes.hpp"

class ARPipeline
{
public:
    ARPipeline(const cv::Mat& patternImage, const CameraCalibration& calibration);
    
    std::vector<cv::Point2f> processFrame(cv::Mat inputFrame);
    
    const PatternTrackingInfo& getPatternLocation() const;
    
    PatternDetector     m_patternDetector;
    Pattern             m_pattern;

private:
    
private:
    CameraCalibration   m_calibration;
    PatternTrackingInfo m_patternInfo;
    //PatternDetector     m_patternDetector;
};

#endif
