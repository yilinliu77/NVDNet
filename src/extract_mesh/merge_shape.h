#pragma once

#include "classify_points_region_growing.h"
#include "kd_tree_helper.h"
#include "shape2d.h"


std::vector<std::shared_ptr<Shape>> merge_shape(std::vector<std::shared_ptr<Shape>>& v_shapes, const double epsilon, const int resolution, const std::string& v_type = "");
std::vector<std::shared_ptr<Shape>> merge_shape(std::vector<std::shared_ptr<Shape>>& v_shapes, const double epsilon, const int resolution, Eigen::MatrixXi& v_adjacent_matrix, const std::string& v_type="");
