#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "CastDataType.h"
#include "tracking_export.h"
#include "Track.h"

class Extractor;

template<typename T>
class TrackerManager;

template<typename T>
class FeatureMetric;

class TRACKING_EXPORT DeepSORT {
public:
    explicit DeepSORT(const std::array<int64_t, 2> &dim, const std::string weight_path);
    explicit DeepSORT(const int width, const int height, const std::string weight_path);

    ~DeepSORT();

    std::vector<std::vector<float>> update(const std::vector<cv::Rect2f> &detections, cv::Mat ori_img);
#ifdef BUILD_PYTHON_PACKAGE
    std::vector<std::vector<float>> updateFromPy(py::array_t<float> x1y1x2y2,
                                    py::array_t<unsigned char>& ori_img);
#endif

private:
    class TrackData;

    std::vector<TrackData> data;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<TrackerManager<TrackData>> manager;
    std::unique_ptr<FeatureMetric<TrackData>> feat_metric;
};


#endif //DEEPSORT_H
