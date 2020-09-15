//
// Created by rainweic on 2020/9/11.
//

#ifndef TRACKING_CASTDATATYPE_H
#define TRACKING_CASTDATATYPE_H

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Track.h"

namespace py = pybind11;

// 类型转换 cv::Rect2f与tuple相互转换
namespace pybind11 {
    namespace detail {
        template <>
        struct type_caster<cv::Rect2f> {
        PYBIND11_TYPE_CASTER(cv::Rect2f, _("list_xywh"));

            bool load(handle obj, bool) {
                if (!py::isinstance<py::list>(obj)) {
                    std::logic_error("Rect2f(x, y, width, height) should be a list!");
                    return false;
                }
                py::list rect2f = reinterpret_borrow<py::list>(obj);
                value = cv::Rect2f(
                        rect2f[0].cast<float>(),
                        rect2f[1].cast<float>(),
                        rect2f[2].cast<float>(),
                        rect2f[3].cast<float>());
                return true;
            }

            static handle cast(const cv::Rect2f &rect, return_value_policy, handle) {
                return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
            }
        };

        template <>
        struct type_caster<Track> {
            PYBIND11_TYPE_CASTER(Track, _("Track_idxywh"));

            bool load(handle obj, bool) {
                py::list track = reinterpret_borrow<py::list>(obj);
                value.id = track[0].cast<int>();
                value.box = cv::Rect2f(
                        track[1].cast<float>(),
                        track[2].cast<float>(),
                        track[3].cast<float>(),
                        track[4].cast<float>()
                        );
                return true;
            }

            static handle cast(const Track &track, return_value_policy, handle) {
                return py::make_tuple(
                        track.id,
                        track.box.x,
                        track.box.y,
                        track.box.width,
                        track.box.height).release();
            }
        };
    }
}

#endif //TRACKING_CASTDATATYPE_H
