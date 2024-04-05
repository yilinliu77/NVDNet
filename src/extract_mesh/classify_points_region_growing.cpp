#include "classify_points_region_growing.h"

#include "tools.h"
#include <cgal_tools.h>


std::pair<std::vector<Eigen::Vector3i>, Eigen::Tensor<bool, 3>> region_growing(
	const Eigen::Tensor<bool, 3>& v_init_visited, const Eigen::Tensor<bool, 3>& v_consistent_flags, const int v_i)
{
	const int resolution = static_cast<int>(v_init_visited.dimension(0));
	std::mt19937_64 gen(v_i);
	std::uniform_int_distribution<> dis(0, resolution - 1);

	Eigen::Tensor<bool, 3> is_visited(v_init_visited);
	int x, y, z;
	while (true)
	{
		x = dis(gen);
		y = dis(gen);
		z = dis(gen);

		if (is_visited(x, y, z))
			continue;
		break;
	}

	std::vector<Eigen::Vector3i> local_clusters;
	std::queue<std::tuple<int, int, int>> queues;
	queues.push({x, y, z});
	while (!queues.empty())
	{
		std::tie(x, y, z) = queues.front();
		queues.pop();
		if (x < 0 || x >= resolution ||
			y < 0 || y >= resolution ||
			z < 0 || z >= resolution ||
			is_visited(x, y, z))
			continue;
		is_visited(x, y, z) = true;
		if (v_consistent_flags(x, y, z))
			continue;
		local_clusters.emplace_back(x, y, z);
		add_neighbours(x, y, z, queues);
	}
	return {local_clusters, is_visited};
}

std::vector<Cluster> classify_points_region_growing(const Eigen::Tensor<bool, 3>& consistent_flags,
                                                    const int resolution, const int num_cpus,
                                                    const Eigen::Tensor<double, 4>& features)
{
	std::vector<Cluster> clusters;
	bool is_log = true;
	{
		Eigen::Tensor<bool, 3> is_visited(consistent_flags);
		LOG(INFO) << ffmt("Start with %d cpus") % num_cpus;

		while (true)
		{
			LOG_IF(INFO, is_log) << "===== New round =====";
			// Grow in parallel
			std::vector<std::vector<Eigen::Vector3i>> local_clusters(num_cpus);
			std::vector<Eigen::Tensor<bool, 3>> local_visited(num_cpus);

#pragma omp parallel for
			for (int i = 0; i < num_cpus; ++i)
			{
				std::tie(local_clusters[i], local_visited[i]) = region_growing(is_visited, consistent_flags, i);
			}

			// Check duplication
			std::vector<bool> is_deleted(local_clusters.size(), false);
			for (int i = 0; i < local_clusters.size(); ++i)
			{
				for (int j = i + 1; j < local_clusters.size(); ++j)
				{
					Eigen::Tensor<bool, 0> a = (local_visited[i] == local_visited[j]).all();
					if (a(0))
						is_deleted[j] = true;
				}
			}

			// Merge
			for (int i = 0; i < local_clusters.size(); ++i)
			{
				if (is_deleted[i])
					continue;
				Cluster cluster;
				cluster.coords = local_clusters[i];
				clusters.emplace_back(cluster);
				is_visited = is_visited || local_visited[i];
			}

			// Check end
			Eigen::Tensor<bool, 0> a = (is_visited == true).all();
			if (a(0))
				break;

			const int num_added = is_deleted.size() - std::accumulate(is_deleted.begin(), is_deleted.end(), 0);
			LOG_IF(INFO, is_log) << ffmt("Done; Add %d clusters; We have %d clusters") % num_added % clusters.size();
		}

		LOG(INFO) << ffmt("We have %d clusters") % clusters.size();
	}
	const int num_clusters = clusters.size();

	// 4. Copy the cluster id
	Eigen::Tensor<int, 3> id_clusters(resolution, resolution, resolution);
	{
		id_clusters.setConstant(-1);
		for (int i_cluster = 0; i_cluster < clusters.size(); ++i_cluster)
		{
			for (const auto& item : clusters[i_cluster].coords)
			{
				const int x = item[0];
				const int y = item[1];
				const int z = item[2];
				id_clusters(x, y, z) = i_cluster;
			}
		}
	}

	// 5. Visualize the clusters
	{
		Point_set total_segmentation(true);
		auto index_map = total_segmentation.add_property_map<int>("index", 0).first;

		int num_valid_clusters = 0;
		for (int i_cluster = 0; i_cluster < clusters.size(); ++i_cluster)
		{
			auto& cluster = clusters[i_cluster];
			const long long num_points = cluster.coords.size();
			if (num_points < 10)
			{
				LOG(INFO) << ffmt("Omit %d points in cluster %d") % num_points % i_cluster;
				continue;
			}

			cluster.query_points.resize(num_points);
			cluster.surface_points.resize(num_points);
			cluster.surface_normals.resize(num_points);
			Point_set pointset;
			for (int i_point = 0; i_point < num_points; ++i_point)
			{
				const Eigen::Vector3i& voxel_coor = clusters[i_cluster].coords[i_point];
				Eigen::Vector3d cur_pos = voxel_coor.cast<double>();
				cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();
				cluster.query_points[i_point] = cur_pos;

				pointset.insert(eigen_2_cgal_point(cur_pos));

				Eigen::Vector3d g_direction(
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 1),
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 2),
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 3)
				);
				g_direction.normalize();

				double dis = features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 0);
				Eigen::Vector3d original_point = cur_pos + g_direction * dis;
				cluster.surface_points[i_point] = original_point;
				cluster.surface_normals[i_point] = -g_direction;
				pointset.insert(eigen_2_cgal_point(original_point));

				index_map[*total_segmentation.insert(eigen_2_cgal_point(original_point))] = num_valid_clusters;
			}
			CGAL::IO::write_point_set((ffmt("temp/cell/p_%d.ply") % i_cluster).str(), pointset);
			num_valid_clusters += 1;
		}
		colorize_point_set(total_segmentation);
		CGAL::IO::write_point_set("temp/summary/init_point_segmentation.ply", total_segmentation);
	}

	return clusters;
}


// #pragma optimize ("", off)
std::pair<std::vector<Eigen::Vector3i>, Eigen::Tensor<bool, 3>> region_growing_with_connectivity(
	const Eigen::Tensor<bool, 3>& v_init_visited, const Eigen::Tensor<bool, 4>& v_connectivity, const int v_i)
{
	const int resolution = static_cast<int>(v_init_visited.dimension(0));
	std::mt19937_64 gen(v_i);
	std::uniform_int_distribution<> dis(0, resolution - 1);

	Eigen::Tensor<bool, 3> is_visited(v_init_visited);
	int x, y, z;
	while (true)
	{
		x = dis(gen);
		y = dis(gen);
		z = dis(gen);

		if (is_visited(x, y, z))
			continue;
		break;
	}

	std::vector<Eigen::Vector3i> local_clusters;
	std::queue<std::tuple<int, int, int>> queues;
	queues.emplace(x, y, z);
	while (!queues.empty())
	{
		std::tie(x, y, z) = queues.front();
		queues.pop();
		if (!check_range(x, y, z, resolution) ||
			is_visited(x, y, z))
			continue;
		is_visited(x, y, z) = true;

		local_clusters.emplace_back(x, y, z);

		int iter = 0;
		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
				for (int dz = -1; dz <= 1; ++dz)
				{
					if (dx == 0 && dy == 0 && dz == 0)
						continue;

					if (v_connectivity(x, y, z, iter))
						queues.emplace(x + dx, y + dy, z + dz);
					iter++;
				}
	}
	return {local_clusters, is_visited};
}


std::vector<Cluster> classify_points_region_growing(
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const Eigen::Tensor<bool, 4>& connectivity,
	const int resolution,
	const int num_cpus,
	const Eigen::Tensor<double, 4>& features,
	const fs::path& v_output,
	const bool only_evaluate
)
{
	checkFolder(v_output/"region_growing");
	std::vector<Cluster> clusters;
	bool is_log = true;
	{
		Eigen::Tensor<bool, 3> is_visited(v_consistent_flags);
		LOG(INFO) << ffmt("Start with %d cpus") % num_cpus;

		while (true)
		{
			LOG_IF(INFO, is_log) << "===== New round =====";
			// Grow in parallel
			std::vector<std::vector<Eigen::Vector3i>> local_clusters(num_cpus);
			std::vector<Eigen::Tensor<bool, 3>> local_visited(num_cpus);

#pragma omp parallel for
			for (int i = 0; i < num_cpus; ++i)
			{
				std::tie(local_clusters[i], local_visited[i]) = region_growing_with_connectivity(
					is_visited, connectivity, i);
			}

			// Check duplication
			std::vector<bool> is_deleted(local_clusters.size(), false);
			for (int i = 0; i < local_clusters.size(); ++i)
			{
				for (int j = i + 1; j < local_clusters.size(); ++j)
				{
					Eigen::Tensor<bool, 0> a = (local_visited[i] == local_visited[j]).all();
					if (a(0))
						is_deleted[j] = true;
				}
			}

			// Merge
			for (int i = 0; i < local_clusters.size(); ++i)
			{
				if (is_deleted[i])
					continue;
				Cluster cluster;
				cluster.coords = local_clusters[i];
				clusters.emplace_back(cluster);
				is_visited = is_visited || local_visited[i];
			}

			// Check end
			Eigen::Tensor<bool, 0> a = (is_visited == true).all();
			if (a(0))
				break;

			const int num_added = is_deleted.size() - std::accumulate(is_deleted.begin(), is_deleted.end(), 0);
			LOG_IF(INFO, is_log) << ffmt("Done; Add %d clusters; We have %d clusters") % num_added % clusters.size();
		}

		LOG(INFO) << ffmt("We have %d clusters") % clusters.size();
	}
	const int num_clusters = clusters.size();

	// 4. Copy the cluster id
	Eigen::Tensor<int, 3> id_clusters(resolution, resolution, resolution);
	{
		id_clusters.setConstant(-1);
		for (int i_cluster = 0; i_cluster < clusters.size(); ++i_cluster)
		{
			for (const auto& item : clusters[i_cluster].coords)
			{
				const int x = item[0];
				const int y = item[1];
				const int z = item[2];
				id_clusters(x, y, z) = i_cluster;
			}
		}
	}

	// 5. Visualize the clusters
	{
		Point_set total_segmentation(true);
		auto index_map = total_segmentation.add_property_map<int>("index", 0).first;

		int num_valid_clusters = 0;
		for (int i_cluster = 0; i_cluster < clusters.size(); ++i_cluster)
		{
			auto& cluster = clusters[i_cluster];
			const long long num_points = cluster.coords.size();
			if (num_points < 10)
			{
				LOG(INFO) << ffmt("Omit %d points in cluster %d") % num_points % i_cluster;
				continue;
			}

			cluster.query_points.resize(num_points);
			cluster.surface_points.resize(num_points);
			cluster.surface_normals.resize(num_points);
			Point_set pointset;
			for (int i_point = 0; i_point < num_points; ++i_point)
			{
				const Eigen::Vector3i& voxel_coor = clusters[i_cluster].coords[i_point];
				Eigen::Vector3d cur_pos = voxel_coor.cast<double>();
				cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();
				cluster.query_points[i_point] = cur_pos;

				pointset.insert(eigen_2_cgal_point(cur_pos));

				Eigen::Vector3d g_direction(
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 1),
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 2),
					features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 3)
				);
				g_direction.normalize();

				double dis = features(voxel_coor[0], voxel_coor[1], voxel_coor[2], 0);
				Eigen::Vector3d original_point = cur_pos + g_direction * dis;
				cluster.surface_points[i_point] = original_point;
				cluster.surface_normals[i_point] = -g_direction;
				pointset.insert(eigen_2_cgal_point(original_point));

				index_map[*total_segmentation.insert(eigen_2_cgal_point(original_point))] = num_valid_clusters;
			}
			if (!only_evaluate)
				CGAL::IO::write_point_set((v_output / "region_growing" / (ffmt("p_%d.ply") % i_cluster).str()).string(), pointset);
			num_valid_clusters += 1;
		}
		if (!only_evaluate)
		{
			colorize_point_set(total_segmentation);
			CGAL::IO::write_point_set((v_output / "summary/init_point_segmentation.ply").string(), total_segmentation);
		}
	}

	return clusters;
}
