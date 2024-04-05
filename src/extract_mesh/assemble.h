#pragma once

#include "shape2d.h"
#include "shape3d.h"

Eigen::MatrixXi assemble(
	std::vector<std::shared_ptr<Shape>>& shapes, 
	const int resolution, 
	const Eigen::Tensor<double, 4>& v_surface_points,
	Point_set& v_boundary,
	const double common_points_threshold,
	const double shape_epsilon,
	const fs::path& v_output,
	const bool debug_viz = false);