#pragma once
#include "cgal_tools.h"

#include "shape2d.h"

Point_set get_boundaries(const std::vector<std::shared_ptr<Shape>>& v_shapes);

void boundary_grow_surface(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Eigen::Tensor<bool, 3>& v_is_voronoi_boundary,
	const bool is_restricted,
	const int max_num_points=-1
);

void boundary_grow_curve(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Eigen::Tensor<bool, 3>& v_is_voronoi
);

void boundary_grow_curve(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Point_set& surface_boundary
);

void boundary_grow_vertex(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid
);

void boundary_grow_restricted(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<double, 4>& features,
	const double udf_threshold
);
