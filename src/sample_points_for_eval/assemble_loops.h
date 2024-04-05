#pragma once

#include "shape2d.h"
#include "shape3d.h"

void solve_vertex(std::vector<std::shared_ptr<Shape>>& v_shapes, const double vertex_threshold,
	const double resolution,
	Eigen::MatrixXi& v_adj_matrix, const fs::path& v_output_dir, bool debug_viz);

void assemble_loops(
	std::vector<std::shared_ptr<Shape>>& v_shapes, 
	const Eigen::Tensor<double, 4>& v_surface_points,
	Point_set& v_boundary);
