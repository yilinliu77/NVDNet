#pragma once

#include "common_util.h"
#include "cgal_tools.h"

#include <CGAL/Linear_cell_complex_for_combinatorial_map.h>
#include <CGAL/Linear_cell_complex_for_generalized_map.h>

// typedef CGAL::Linear_cell_complex_for_combinatorial_map<3, 3> LCC_3;
typedef CGAL::Linear_cell_complex_for_generalized_map<3, 3> LCC_3;


/*
 * Utils
 */

std::pair<Eigen::Tensor<bool, 3>, Eigen::VectorXi> mark_boundary(const Eigen::Tensor<bool, 3>& v_flags, const Eigen::Tensor<double, 4>& v_udf, const double v_distance_threshold);

Eigen::Tensor<bool, 3> dilate(const Eigen::Tensor<bool, 3>& v_flag, const int resolution, const int half_window_size);

Eigen::Tensor<bool, 3> voxelize(const int resolution, const std::vector<Triangle_3>& v_triangles);

/*
 * Alpha shape related
 */

std::vector<Triangle_3> filling_holes(
	const Point_set& boundary_points, const double alpha_value = 0.0025);

void build_lcc(
	const std::vector<Point_3>& v_points, const std::vector<std::vector<unsigned long long>>& v_faces,
	LCC_3& lcc,
	std::vector<CGAL::Polyhedron_3<K>>& polys,
	bool is_viz_cells
);

Eigen::Tensor<bool, 3> filling_hole(const Eigen::Tensor<bool, 3>& v_flag, const Eigen::Tensor<double, 4>& v_gradient, const int resolution);

Eigen::Tensor<bool, 3> rebuild_flags(const std::vector<Triangle_3>& v_triangles,
	const Eigen::Tensor<bool, 3>& v_edge_flag, const int v_num_per_m2 = 256 * 256 * 10);

/*
 * Alpha shape related
 */

Eigen::Tensor<bool, 3> dilate_according_to_gradients(
	const Eigen::Tensor<bool, 3>& v_flag,
	const Eigen::Tensor<double, 4>& v_features,
	const int window_size, const double v_udf_threshold
);

Eigen::Tensor<bool, 3> dilate_along_the_ray(
	const Eigen::Tensor<double, 4>& v_features,
	const double v_udf_threshold,
	const Eigen::Tensor<bool, 3>& v_flags,
	const double threshold
);

Eigen::Tensor<bool, 4> build_edge_connectivity(
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const int half_window_size
);