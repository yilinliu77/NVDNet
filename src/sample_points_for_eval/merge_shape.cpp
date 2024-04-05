#include "merge_shape.h"

#include "fitting.h"
#include "shape3d.h"

#include <tbb/tbb.h>

// #pragma optimize ("", off)

bool merge_items(std::vector<std::shared_ptr<Shape>>& v_shapes, const int i_shape, const int i_shape2, const double epsilon)
{
	if (v_shapes[i_shape]->inliers.empty() || v_shapes[i_shape2]->inliers.empty())
		return false;

	double error = 0;
	Point_set p2;
	for (int i_point2 = 0; i_point2 < v_shapes[i_shape2]->inliers.size(); ++i_point2)
	{
		error += v_shapes[i_shape]->distance(v_shapes[i_shape2]->inliers[i_point2]);
		p2.insert(eigen_2_cgal_point(v_shapes[i_shape2]->inliers[i_point2]));
	}
	error /= v_shapes[i_shape2]->inliers.size();
	// LOG(INFO) << error;
	// CGAL::IO::write_point_set("temp/p2.ply", p2);
	if (error < epsilon)
	{
		LOG(INFO) << "Merging " << i_shape2 << " into " << i_shape << " with error " << error;
		// CGAL::IO::write_point_set("temp/p2.ply", p2);

		Cluster cluster = v_shapes[i_shape]->cluster;
		cluster.surface_points.insert(
			cluster.surface_points.end(),
			v_shapes[i_shape2]->cluster.surface_points.begin(),
			v_shapes[i_shape2]->cluster.surface_points.end());
		cluster.coords.insert(
			cluster.coords.end(),
			v_shapes[i_shape2]->cluster.coords.begin(),
			v_shapes[i_shape2]->cluster.coords.end());
		cluster.query_points.insert(
			cluster.query_points.end(),
			v_shapes[i_shape2]->cluster.query_points.begin(),
			v_shapes[i_shape2]->cluster.query_points.end());
		cluster.surface_normals.insert(
			cluster.surface_normals.end(),
			v_shapes[i_shape2]->cluster.surface_normals.begin(),
			v_shapes[i_shape2]->cluster.surface_normals.end());

		// std::vector<Eigen::Vector3d> inliers = v_shapes[i_shape]->inliers;
		// inliers.insert(
		// 	inliers.end(),
		// 	v_shapes[i_shape2]->inliers.begin(),
		// 	v_shapes[i_shape2]->inliers.end());

		std::pair<std::shared_ptr<Shape>, double> result;
		if (v_shapes[i_shape]->type == "curve")
			result = fit_curve(cluster.surface_points, cluster, dynamic_pointer_cast<Shape2D>(v_shapes[i_shape])->m_plane, v_shapes[i_shape]->detail_type);
		else if (v_shapes[i_shape]->type == "surface")
			result = fit_surface(cluster.surface_points, cluster, v_shapes[i_shape]->detail_type);
		else
			result = fit_vertex(cluster.surface_points, cluster, epsilon);

		if (result.first == nullptr)
		{
			LOG(ERROR) << "Re fit error when merging items";
			return false;
		}


		result.first->get_inliers(cluster.surface_points, epsilon);

		result.first = check_valid_ellipse(result.first, result.first->cluster.surface_points);
		if (result.first == nullptr)
			return false;

		if (result.first->detail_type == "ellipse" && dynamic_pointer_cast<MyEllipse>(result.first)->ellipse.MajorRadius() > 1.)
			return false;

		if (result.first->inliers.empty())
			return false;

		v_shapes[i_shape] = result.first;
		v_shapes[i_shape]->get_inliers(cluster.surface_points, epsilon);
		v_shapes[i_shape]->find_boundary();
		LOG(INFO) << ffmt("Re-fit error: %d") % result.second;
		return true;
	}
	else
		return false;
}

std::vector<std::shared_ptr<Shape>> merge_shape(
	std::vector<std::shared_ptr<Shape>>& v_shapes, const double epsilon, const int resolution,
	const std::string& v_type)
{
	std::vector<std::shared_ptr<my_kd_tree_t>> kdtrees(v_shapes.size(), nullptr);

	const double inlier_epsilon = 1e-2; // 0.007 * 2

	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (!v_type.empty() && v_shapes[i_shape]->type != v_type)
					continue;

				Point_set points;
				points.resize(v_shapes[i_shape]->inliers.size());
				for (int i = 0; i < points.size(); ++i)
					points.point(i) = eigen_2_cgal_point(v_shapes[i_shape]->inliers[i]);
				kdtrees[i_shape].reset(initialize_kd_tree(points));
			}
		});
	

	Eigen::MatrixXi adjacency_count(v_shapes.size(), v_shapes.size());
	adjacency_count.setZero();

	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (!v_type.empty() && v_shapes[i_shape]->type != v_type)
					continue;

				for (int i_point = 0; i_point < v_shapes[i_shape]->inliers.size(); ++i_point)
					for (int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
					{
						if (!v_type.empty() && v_shapes[i_shape2]->type != v_type)
							continue;
						if (i_shape == i_shape2 || v_shapes[i_shape]->type != v_shapes[i_shape2]->type)
							continue;
						auto search_result = search_range(*kdtrees[i_shape2], v_shapes[i_shape]->inliers[i_point].cast<float>(), inlier_epsilon);
						if (!search_result.empty())
							adjacency_count(i_shape, i_shape2) += 1;
					}
			}
		});
	std::vector<bool> deleted_flag(v_shapes.size(), false);
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		if (!v_type.empty() && v_shapes[i_shape]->type != v_type)
			continue;
		if (deleted_flag[i_shape])
			continue;
		Point_set p1;

		std::vector<bool> visited_flags(v_shapes.size(), false);
		for (int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
			if (v_shapes[i_shape]->type != v_shapes[i_shape2]->type || i_shape == i_shape2 || deleted_flag[i_shape2])
				visited_flags[i_shape2] = true;

		std::queue<int> related_ids;
		for(int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
		{
			if (!v_type.empty() && v_shapes[i_shape2]->type != v_type)
				continue;
			if (visited_flags[i_shape2])
				continue;

			// Check if i_shape2 can be merged into i_shape1
			if (adjacency_count(i_shape, i_shape2) > 0)
			{
				visited_flags[i_shape2] = true;
				if (merge_items(v_shapes, i_shape, i_shape2, epsilon))
				{
					for (int i_shape3 = 0; i_shape3 < v_shapes.size(); ++i_shape3)
					{
						if (visited_flags[i_shape3])
							continue;
						if (adjacency_count(i_shape2, i_shape3) > 0)
							related_ids.emplace(i_shape3);
					}
					deleted_flag[i_shape2] = true;
				}
			}
		}

		while (!related_ids.empty())
		{
			const int i_shape2 = related_ids.front();
			related_ids.pop();
			if(visited_flags[i_shape2])
				continue;
			visited_flags[i_shape2] = true;
			if (merge_items(v_shapes, i_shape, i_shape2, epsilon))
			{
				for (int i_shape3 = 0; i_shape3 < v_shapes.size(); ++i_shape3)
				{
					if (!v_type.empty() && v_shapes[i_shape3]->type != v_type)
						continue;
					if (visited_flags[i_shape3])
						continue;
					if (adjacency_count(i_shape2, i_shape3) > 0)
						related_ids.emplace(i_shape3);
				}
				deleted_flag[i_shape2] = true;
			}
		}
		continue;
	}

	std::vector<std::shared_ptr<Shape>> merged_shapes;
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		if (v_type.empty())
		{
			if (!deleted_flag[i_shape])
				merged_shapes.push_back(v_shapes[i_shape]);
		}
		else
		{
			if (v_shapes[i_shape]->type != v_type)
				merged_shapes.push_back(v_shapes[i_shape]);
			else if (!deleted_flag[i_shape])
				merged_shapes.push_back(v_shapes[i_shape]);
		}
	}

	return merged_shapes;
}

std::vector<std::shared_ptr<Shape>> merge_shape(std::vector<std::shared_ptr<Shape>>& v_shapes, const double epsilon, const int resolution, Eigen::MatrixXi& v_adjacent_matrix, const std::string& v_type)
{
	bool use_detail_type = false;
	if (v_type == "curve")
		use_detail_type = true;
	std::vector<std::shared_ptr<my_kd_tree_t>> kdtrees(v_shapes.size(), nullptr);

	const double inlier_epsilon = 1e-2; // 0.007 * 2

	#pragma omp parallel for
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		Point_set points;
		points.resize(v_shapes[i_shape]->inliers.size());
		for (int i = 0; i < points.size(); ++i)
			points.point(i) = eigen_2_cgal_point(v_shapes[i_shape]->inliers[i]);
		kdtrees[i_shape].reset(initialize_kd_tree(points));
	}

	Eigen::MatrixXi adjacency_count(v_shapes.size(), v_shapes.size());
	adjacency_count.setZero();

	#pragma omp parallel for
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		for (int i_point = 0; i_point < v_shapes[i_shape]->inliers.size(); ++i_point)
			for (int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
			{
				if (i_shape == i_shape2 || v_shapes[i_shape]->type != v_shapes[i_shape2]->type)
					continue;
				auto search_result = search_range(*kdtrees[i_shape2], v_shapes[i_shape]->inliers[i_point].cast<float>(), inlier_epsilon);
				if (!search_result.empty())
					adjacency_count(i_shape, i_shape2) += 1;
			}
	}

	std::vector<bool> deleted_flag(v_shapes.size(), false);
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		if (!v_type.empty() && v_shapes[i_shape]->type != v_type)
			continue;

		if (deleted_flag[i_shape])
			continue;
		Point_set p1;

		std::vector<bool> visited_flags(v_shapes.size(), false);
		for (int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
		{
			if (use_detail_type)
			{
				if (v_shapes[i_shape]->detail_type != v_shapes[i_shape2]->detail_type ||
					i_shape == i_shape2 ||
					deleted_flag[i_shape2])
					visited_flags[i_shape2] = true;
			}
			else
			{
				if (v_shapes[i_shape]->type != v_shapes[i_shape2]->type ||
					i_shape == i_shape2 ||
					deleted_flag[i_shape2])
					visited_flags[i_shape2] = true;
			}
		}
			

		std::queue<int> related_ids;
		for (int i_shape2 = 0; i_shape2 < v_shapes.size(); ++i_shape2)
		{
			if (visited_flags[i_shape2])
				continue;

			// Check if i_shape2 can be merged into i_shape1
			if (adjacency_count(i_shape, i_shape2) > 0)
			{
				visited_flags[i_shape2] = true;
				if (merge_items(v_shapes, i_shape, i_shape2, epsilon))
				{
					for (int i_shape3 = 0; i_shape3 < v_shapes.size(); ++i_shape3)
					{
						if (adjacency_count(i_shape2, i_shape3) > 0 && !visited_flags[i_shape3])
							related_ids.emplace(i_shape3);
						if (v_adjacent_matrix(i_shape, i_shape3) == 0 && v_adjacent_matrix(i_shape2, i_shape3) != 0)
						{
							v_adjacent_matrix(i_shape, i_shape3) = v_adjacent_matrix(i_shape2, i_shape3);
							v_adjacent_matrix(i_shape3, i_shape) = v_adjacent_matrix(i_shape3, i_shape2);
						}
					}
					deleted_flag[i_shape2] = true;
				}
			}
		}

		while (!related_ids.empty())
		{
			const int i_shape2 = related_ids.front();
			related_ids.pop();
			if (visited_flags[i_shape2])
				continue;
			visited_flags[i_shape2] = true;
			if (merge_items(v_shapes, i_shape, i_shape2, epsilon))
			{
				for (int i_shape3 = 0; i_shape3 < v_shapes.size(); ++i_shape3)
				{
					if (visited_flags[i_shape3])
						continue;
					if (adjacency_count(i_shape2, i_shape3) > 0)
						related_ids.emplace(i_shape3);
					if (v_adjacent_matrix(i_shape, i_shape3) == 0 && v_adjacent_matrix(i_shape2, i_shape3) != 0)
					{
						v_adjacent_matrix(i_shape, i_shape3) = v_adjacent_matrix(i_shape2, i_shape3);
						v_adjacent_matrix(i_shape3, i_shape) = v_adjacent_matrix(i_shape3, i_shape2);
					}
				}
				deleted_flag[i_shape2] = true;
			}
		}
		continue;
	}

	std::vector<std::shared_ptr<Shape>> merged_shapes;
	std::vector<int> indice_to_keep;
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		if (deleted_flag[i_shape])
			continue;
		merged_shapes.push_back(v_shapes[i_shape]);
		indice_to_keep.push_back(i_shape);
	}
	std::sort(indice_to_keep.begin(), indice_to_keep.end());
	Eigen::MatrixXi t = v_adjacent_matrix(indice_to_keep, indice_to_keep);
	v_adjacent_matrix = t;
	return merged_shapes;
}

// #pragma optimize ("", off)
// #pragma optimize ("", on)
