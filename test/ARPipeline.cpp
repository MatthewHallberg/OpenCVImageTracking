#include "ARPipeline.hpp"

ARPipeline::ARPipeline(const cv::Mat& patternImage, const CameraCalibration& calibration)
: m_calibration(calibration)
{
    m_patternDetector.buildPatternFromImage(patternImage, m_pattern);
    m_patternDetector.train(m_pattern);
}

std::vector<cv::Point2f> ARPipeline::processFrame(cv::Mat inputFrame)
{
    std::vector<cv::Point2f> objectPoints = m_patternDetector.findPattern(inputFrame, m_patternInfo);
    
    if (objectPoints.size() > 0)
    {
        m_patternInfo.computePose(m_pattern, m_calibration);
    }
    
    return objectPoints;
}

const Transformation& ARPipeline::getPatternLocation() const
{
    return m_patternInfo.pose3d;
}
