#pragma once
#include <nanoflann.hpp>
#include "model_tools.h"

using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<matrix_t>;

class KdTreeHelper
{
public:
	matrix_t points_vec;
	std::shared_ptr<my_kd_tree_t> kdtree;

	KdTreeHelper(const Point_set& v_points)
	{
		points_vec.resize(v_points.size(), 3);
		for (int i = 0; i < points_vec.rows(); ++i)
			points_vec.row(i) = cgal_point_2_eigen(v_points.point(i));
		kdtree.reset(new my_kd_tree_t(3 /*dim*/, points_vec, {10 /* max leaf */}));
		kdtree->index_->buildIndex();
	}

	KdTreeHelper(const std::vector<Eigen::Vector3d>& v_points)
	{
		points_vec.resize(v_points.size(), 3);
		for (int i = 0; i < points_vec.rows(); ++i)
			points_vec.row(i) = v_points[i].cast<float>();
		kdtree.reset(new my_kd_tree_t(3 /*dim*/, points_vec, {10 /* max leaf */}));
		kdtree->index_->buildIndex();
	}

	std::pair<std::vector<size_t>, std::vector<float>> search_k_neighbour(const Eigen::Vector3f& v_point, const int k)
	{
		const size_t num_results = k;
		std::vector<size_t> ret_indexes(num_results);
		std::vector<float> out_dists_sqr(num_results);

		nanoflann::KNNResultSet<float> resultSet(num_results);

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		kdtree->index_->findNeighbors(
			resultSet, &v_point.x(), nanoflann::SearchParameters());

		return std::make_pair(ret_indexes, out_dists_sqr);
	}

	std::vector<nanoflann::ResultItem<Eigen::Index, float>> search_range(const Eigen::Vector3f& v_point, const float r)
	{
		std::vector<nanoflann::ResultItem<Eigen::Index, float>> resultSet;
		kdtree->index_->radiusSearch(&v_point.x(), r * r, resultSet);
		return resultSet;
	}
};

matrix_t initialize_kd_data(const Point_set& v_points);
matrix_t initialize_kd_data(const std::vector<Eigen::Vector3d>& v_points);
std::shared_ptr<my_kd_tree_t> initialize_kd_tree(const matrix_t& v_data);

my_kd_tree_t* initialize_kd_tree(const Point_set& v_points);

std::pair<std::vector<size_t>, std::vector<float>> search_k_neighbour(const my_kd_tree_t& v_kdtree,
                                                                      const Eigen::Vector3f& v_point, const int k = 1);
std::vector<nanoflann::ResultItem<Eigen::Index, float>> search_range(my_kd_tree_t& v_kdtree,
                                                                     const Eigen::Vector3f& v_point, const float r);
