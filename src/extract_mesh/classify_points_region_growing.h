#pragma once


#include "common_util.h"
#include <unsupported/Eigen/CXX11/Tensor>


#include "shape2d.h"


void add_neighbours(const int x, const int y, const int z,
	std::queue<std::tuple<int, int, int>>& v_queues);
void add_neighbours(const int x, const int y, const int z,
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::queue<Eigen::Vector3i>& v_queues,
	const double v_epsilon);

Eigen::Tensor<bool, 4> build_edge_connectivity(
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const int half_window_size = 2
);

std::pair<std::vector<Eigen::Vector3i>, Eigen::Tensor<bool, 3>> region_growing(
	const Eigen::Tensor<bool, 3>& v_init_visited,
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const int v_i);

std::vector<Cluster> classify_points_region_growing(
	const Eigen::Tensor<bool, 3>& consistent_flags,
	const int resolution,
	const int num_cpus,
	const Eigen::Tensor<double, 4>& features
);

std::vector<Cluster> classify_points_region_growing(
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const Eigen::Tensor<bool, 4>& connectivity,
	const int resolution,
	const int num_cpus,
	const Eigen::Tensor<double, 4>& features,
	const fs::path& v_output,
	const bool only_evaluate
);