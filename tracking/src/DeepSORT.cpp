#include <algorithm>

#include "DeepSORT.h"
#include "Extractor.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

struct DeepSORT::TrackData {
    KalmanTracker kalman;
    FeatureBundle feats;
};

DeepSORT::DeepSORT(const array<int64_t, 2> &dim)
        : extractor(make_unique<Extractor>()),
          manager(make_unique<TrackerManager<TrackData>>(data, dim)),
          feat_metric(make_unique<FeatureMetric<TrackData>>(data)) {}

DeepSORT::DeepSORT(const int width, const int height) {
    array<int64_t, 2> ori_dim{int64_t(height), int64_t(width)};
    new (this) DeepSORT(ori_dim);
}

DeepSORT::~DeepSORT() = default;

vector<vector<float>> DeepSORT::update(const std::vector<cv::Rect2f> &detections, cv::Mat ori_img) {
    manager->predict();
    manager->remove_nan();

    auto matched = manager->update(
            detections,
            [this, &detections, &ori_img](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(data[t].kalman.rect());
                }
                vector<cv::Mat> boxes;
                vector<cv::Rect2f> dets;
                for (auto d:det_ids) {
                    dets.push_back(detections[d]);
                    boxes.push_back(ori_img(detections[d]));
                }

                auto iou_mat = iou_dist(dets, trks);
                auto feat_mat = feat_metric->distance(extractor->extract(boxes), trk_ids);
                feat_mat.masked_fill_((iou_mat > 0.8f).__ior__(feat_mat > 0.2f), INVALID_DIST);
                return feat_mat;
            },
            [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(data[t].kalman.rect());
                }
                vector<cv::Rect2f> dets;
                for (auto &d:det_ids) {
                    dets.push_back(detections[d]);
                }
                auto iou_mat = iou_dist(dets, trks);
                iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
                return iou_mat;
            });

    vector<cv::Mat> boxes;
    vector<int> targets;
    for (auto[x, y]:matched) {
        targets.emplace_back(x);
        boxes.emplace_back(ori_img(detections[y]));
    }
    feat_metric->update(extractor->extract(boxes), targets);

    manager->remove_deleted();

    return manager->visible_tracks();
}

#ifdef BUILD_PYTHON_PACKAGE
vector<vector<float>> DeepSORT::updateFromPy(py::array_t<float> x1y1x2y2,
                                     py::array_t<unsigned char>& ori_img) {
    std::vector<cv::Rect2f> detestions;
    py::buffer_info x1y1x2y2_buf = x1y1x2y2.request();

    if (x1y1x2y2_buf.ndim != 2)
        throw std::runtime_error("输入的物体的Bbox必须是2维的ndarray");

    float *x1y1x2y2_ptr = (float *) x1y1x2y2_buf.ptr;

    for (size_t i = 0; i < x1y1x2y2_buf.shape[0]; i++) {
        float x1 = x1y1x2y2_ptr[i*4 + 0];
        float y1 = x1y1x2y2_ptr[i*4 + 1];
        float width = x1y1x2y2_ptr[i*4 + 2] - x1;
        float height = x1y1x2y2_ptr[i*4 + 3] - y1;
        detestions.push_back(cv::Rect2f{x1, y1, width, height});
    }

    py::buffer_info buf = ori_img.request();

    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return this->update(detestions, img);
}


PYBIND11_MODULE(tracking, m) {
    py::class_<DeepSORT>(m, "deepsort")
            .def(py::init<int, int>())
            .def("update", &DeepSORT::updateFromPy);
}
#endif
