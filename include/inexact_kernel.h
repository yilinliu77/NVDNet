#pragma once

#include "common_util.h"

#include <glog/logging.h>
#include "model_tools.h"
#include "cgal_tools.h"

const double NUMERICAL_THRESHOLD = 1e-8;

// cos(1e-3 degree) = 0.99999999984769129011051202417815
template<typename A1>
bool is_parallel(const A1& v1, const A1& v2, const double v_eps = 0.99999999984769129011051202417815);
// cos(1e-3 degree) = 0.99999999984769129011051202417815
bool is_colinear(const Point_2& p1, const Point_2& p2, const Point_2& p3, const double v_eps = 0.99999999984769129011051202417815);

// cos(1e-3 degree) = 0.99999999984769129011051202417815
bool is_same_direction(const Vector_2& v1, const Vector_2& v2, const double v_eps = 0.99999999984769129011051202417815);

// cos(1e-3 degree) = 0.99999999984769129011051202417815
bool is_oppsite_direction(const Vector_2& v1, const Vector_2& v2, const double v_eps = 0.99999999984769129011051202417815);

bool is_overlap(const Line_2& v_line1, const Line_2& v_line2);

template <typename A1>
bool is_overlap(const A1& v_point1, const A1& v_point2, const double v_eps = NUMERICAL_THRESHOLD);

template<typename A1, typename A2, typename A3>
std::pair<bool, A3> intersect(const A1& item1, const A2& item2);

std::pair<bool, Point_2> intersect_(const Line_2& item1, const Segment_2& item2);

std::pair<bool, Point_2> intersect_(const Segment_2& item1, const Line_2& item2);

std::pair<bool, Point_2> intersect_(const Line_2& item1, const Ray_2& item2);

std::pair<bool, std::vector<Point_2>> intersect_(const Line_2& item1, const Polygon_2& item2);

