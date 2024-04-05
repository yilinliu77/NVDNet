#include "kd_tree_helper.h"

matrix_t initialize_kd_data(const Point_set& v_points)
{
	matrix_t points_vec(v_points.size(), 3);
	int cur_index = 0;
	for (const auto& item : v_points)
	{
		points_vec.row(cur_index) = cgal_point_2_eigen(v_points.point(item));
		cur_index += 1;
	}
	return points_vec;
}

matrix_t initialize_kd_data(const std::vector<Eigen::Vector3d>& v_points)
{
	matrix_t points_vec(v_points.size(), 3);
	int cur_index = 0;
	for (const auto& item : v_points)
	{
		points_vec.row(cur_index) = item.cast<float>();
		cur_index += 1;
	}
	return points_vec;
}

std::shared_ptr<my_kd_tree_t> initialize_kd_tree(const matrix_t& v_data)
{
	std::shared_ptr<my_kd_tree_t> kdtree(new my_kd_tree_t(3 /*dim*/, v_data, { 10 /* max leaf */ }));
	kdtree->index_->buildIndex();
	return kdtree;
}

my_kd_tree_t* initialize_kd_tree(const Point_set& v_points)
{
	// matrix_t points_vec(v_points.size(), 3);
	matrix_t* points_vec = new matrix_t(v_points.size(), 3);
	points_vec->resize(v_points.size(), 3);
	int cur_index = 0;
	for (const auto& item : v_points)
	{
		points_vec->row(cur_index) = cgal_point_2_eigen(v_points.point(item));
		cur_index += 1;
	}
	my_kd_tree_t* kdtree = new my_kd_tree_t(3 /*dim*/, *points_vec, { 10 /* max leaf */ });
	
	kdtree->index_->buildIndex();
	return kdtree;
}

std::pair<std::vector<size_t>, std::vector<float>> search_k_neighbour(const my_kd_tree_t& v_kdtree,
	const Eigen::Vector3f& v_point, const int k)
{
	const size_t        num_results = k;
	std::vector<size_t> ret_indexes(num_results);
	std::vector<float>  out_dists_sqr(num_results);

	nanoflann::KNNResultSet<float> resultSet(num_results);

	resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
	v_kdtree.index_->findNeighbors(
		resultSet, &v_point.x(), nanoflann::SearchParameters());

	return std::make_pair(ret_indexes, out_dists_sqr);

}

std::vector<nanoflann::ResultItem<Eigen::Index, float>> search_range(my_kd_tree_t& v_kdtree,
	const Eigen::Vector3f& v_point, const float r)
{
	std::vector<nanoflann::ResultItem<Eigen::Index, float>> resultSet;
	v_kdtree.index_->radiusSearch(&v_point.x(), r * r, resultSet);

	return resultSet;

}