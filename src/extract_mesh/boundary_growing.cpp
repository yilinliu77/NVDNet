#include "common_util.h"

#include "boundary_growing.h"

#include "classify_points_region_growing.h"
#include "fitting.h"
#include "kd_tree_helper.h"
#include "shape3d.h"
#include "tools.h"

#include <tbb/tbb.h>

int flat_voxel_coord(const int x, const int y, const int z, const int res)
{
	return x * res * res + y * res + z;
}

std::tuple<int,int,int> deflat_voxel_coord(const int x, const int res)
{
	return { x / res / res, x / res % res,x % res };
}

Point_set get_boundaries(const std::vector<std::shared_ptr<Shape>>& v_shapes)
{
	Point_set boundaries;
	for (int i_shape = 0; i_shape < v_shapes.size(); ++i_shape)
	{
		if (v_shapes[i_shape]->type != "surface")
			continue;
		for (const auto& item : v_shapes[i_shape]->boundary_points)
			boundaries.insert(eigen_2_cgal_point(item));
	}
	return boundaries;
}

// #pragma optimize ("", off)

void boundary_grow_surface(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Eigen::Tensor<bool, 3>& v_is_voronoi_boundary,
	const bool is_restricted,
	const int max_num_points
)
{
	const int resolution = v_surface_points.dimension(0);
	const double nearby_threshold = 0.05; // Do not expand if the nearby surface point is too far

	// For Surface
	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (shapes[i_shape]->type != "surface")
					continue;

				Eigen::Tensor<bool, 3> is_valid ;
				if (is_restricted)
					is_valid = v_is_valid && v_is_voronoi_boundary;
				else
					is_valid = v_is_valid;
				std::queue<Eigen::Vector3i> queues;
				// Mark all the existing coordinate to false
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					is_valid(c[0], c[1], c[2]) = false;
				}
				// Initialize the queue using 26 neighbour
				// Skip the neighbour which is too far
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								if (dx == 0 && dy == 0 && dz == 0)
									continue;
								const int nx = c[0] + dx;
								const int ny = c[1] + dy;
								const int nz = c[2] + dz;
								if (!check_range(nx, ny, nz, resolution))
									continue;
								if (!is_valid(nx, ny, nz))
									continue;
								is_valid(nx, ny, nz) = false;
								const Eigen::Vector3d p1(
									v_surface_points(c[0], c[1], c[2], 0),
									v_surface_points(c[0], c[1], c[2], 1),
									v_surface_points(c[0], c[1], c[2], 2));
								const Eigen::Vector3d p2(
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 0),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 1),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 2));
								if ((p1 - p2).norm() < nearby_threshold)
									queues.emplace(c[0] + dx, c[1] + dy, c[2] + dz);
							}
				}
				// Iterate the queues
				// For valid expansion, add the coordinate to the cluster.coords, cluster.surface_points, inliers
				while (!queues.empty())
				{
					const auto c = queues.front();
					queues.pop();

					Eigen::Vector3d cur_point(
						v_surface_points(c[0], c[1], c[2], 0),
						v_surface_points(c[0], c[1], c[2], 1),
						v_surface_points(c[0], c[1], c[2], 2)
					);
					if (shapes[i_shape]->distance(cur_point) < epsilon)
					{
						is_valid(c[0], c[1], c[2]) = false;
						shapes[i_shape]->cluster.coords.emplace_back(c);
						shapes[i_shape]->cluster.surface_points.emplace_back(cur_point);
						shapes[i_shape]->inliers.emplace_back(cur_point);
						for (int dx = -1; dx <= 1; dx++)
							for (int dy = -1; dy <= 1; dy++)
								for (int dz = -1; dz <= 1; dz++)
								{
									if (dx == 0 && dy == 0 && dz == 0)
										continue;
									const int nx = c[0] + dx;
									const int ny = c[1] + dy;
									const int nz = c[2] + dz;
									if (!check_range(nx, ny, nz, resolution))
										continue;
									if (!is_valid(nx, ny, nz))
										continue;
									const Eigen::Vector3d p1(
										v_surface_points(c[0], c[1], c[2], 0),
										v_surface_points(c[0], c[1], c[2], 1),
										v_surface_points(c[0], c[1], c[2], 2));
									const Eigen::Vector3d p2(
										v_surface_points(nx, ny, nz, 0),
										v_surface_points(nx, ny, nz, 1),
										v_surface_points(nx, ny, nz, 2));
									is_valid(nx, ny, nz) = false;
									if ((p1 - p2).norm() < nearby_threshold)
										queues.emplace(nx, ny, nz);
								}
					}
				}
			}
		});

	// One position can only has one shape
	// Assign the shape id and coord id to each voxel
	std::vector < std::vector < Eigen::Vector2i >> voxel_to_shape(resolution* resolution* resolution);
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
		{
			const auto& coord = shapes[i_shape]->cluster.coords[i_coord];
			voxel_to_shape[flat_voxel_coord(coord[0], coord[1], coord[2], resolution)].emplace_back(
				i_shape, i_coord);
		}

	// Initialize the marking array
	std::vector<std::vector<bool>> is_deleted(shapes.size());
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		is_deleted[i_shape].resize(shapes[i_shape]->cluster.surface_points.size(), false);

	// Iter each voxel, compute the distance and mark the invalid voxel
	for (int i=0;i<256*256*256;++i)
	{
		if (voxel_to_shape[i].size() <= 1)
			continue;

		auto [x, y, z] = deflat_voxel_coord(i, resolution);

		const Eigen::Vector3d surface_point(
			v_surface_points(x,y,z,0),
			v_surface_points(x,y,z,1),
			v_surface_points(x,y,z,2)
		);
		std::vector<double> distance(voxel_to_shape[i].size());
		for (int j=0;j< voxel_to_shape[i].size();++j)
			distance[j] = shapes[voxel_to_shape[i][j][0]]->distance(surface_point);
		const int min_idx = std::min_element(distance.begin(), distance.end()) - distance.begin();
		for(int j=0;j< voxel_to_shape[i].size();++j)
			if (j != min_idx)
			{
				const int i_shape = voxel_to_shape[i][j][0];
				is_deleted[i_shape][voxel_to_shape[i][j][1]] = true;
			}
	}

	// Perform deletion and re fit surface
	for (int i_shape = shapes.size() - 1; i_shape >= 0; --i_shape)
	{
		if (shapes[i_shape]->type != "surface")
			continue;
		auto& surface_points = shapes[i_shape]->cluster.surface_points;
		surface_points.erase(
			std::remove_if(surface_points.begin(), surface_points.end(),
								[&is_deleted, &surface_points, &i_shape](const auto& item)
								{
									return is_deleted[i_shape][&item-&surface_points[0]];
								}),
			surface_points.end());
		auto& coords = shapes[i_shape]->cluster.coords;
		coords.erase(
			std::remove_if(coords.begin(), coords.end(),
				[&is_deleted, &coords, &i_shape](const auto& item)
				{
					return is_deleted[i_shape][&item - &coords[0]];
				}),
			coords.end());

		if (max_num_points < 0)
			shapes[i_shape] = fit_surface(
				shapes[i_shape]->cluster.surface_points,
				shapes[i_shape]->cluster,
				shapes[i_shape]->detail_type).first;
		else
		{
			std::vector<gte::Vector3<double>> data1;
			prepare_gte_data(shapes[i_shape]->inliers, data1, max_num_points);
			shapes[i_shape] = fit_surface(
				data1,
				shapes[i_shape]->cluster,
				shapes[i_shape]->detail_type).first;
		}
		

		if (shapes[i_shape] == nullptr)
		{
			LOG(ERROR) << "refit surface failed in boundary growing";
			shapes.erase(shapes.begin() + i_shape);
			continue;
		}

		shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
		shapes[i_shape]->find_boundary();
	}
	
	return;
}

void boundary_grow_curve(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Eigen::Tensor<bool, 3>& v_is_voronoi
)
{
	const double distance_to_boundary = 1e-3;
	const double nearby_threshold = 1e-2;
	const int res = v_surface_points.dimension(0);

	Eigen::Tensor<bool, 3> is_valid_global = v_is_valid && v_is_voronoi;

	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (shapes[i_shape]->type != "curve")
					continue;

				Eigen::Tensor<bool, 3> is_valid = is_valid_global;
				std::queue<Eigen::Vector3i> queues;
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					is_valid(c[0], c[1], c[2]) = false;
				}
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								if (dx == 0 && dy == 0 && dz == 0)
									continue;
								const int nx = c[0] + dx;
								const int ny = c[1] + dy;
								const int nz = c[2] + dz;
								if (!check_range(nx, ny, nz, res))
									continue;
								if (!is_valid(nx, ny, nz))
									continue;
								is_valid(nx, ny, nz) = false;
								const Eigen::Vector3d p1(
									v_surface_points(c[0], c[1], c[2], 0),
									v_surface_points(c[0], c[1], c[2], 1),
									v_surface_points(c[0], c[1], c[2], 2));
								const Eigen::Vector3d p2(
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 0),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 1),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 2));
								if ((p1 - p2).norm() < nearby_threshold)
									queues.emplace(c[0] + dx, c[1] + dy, c[2] + dz);
							}
				}
				while (!queues.empty())
				{
					const auto c = queues.front();
					queues.pop();

					Eigen::Vector3d cur_point(
						v_surface_points(c[0], c[1], c[2], 0),
						v_surface_points(c[0], c[1], c[2], 1),
						v_surface_points(c[0], c[1], c[2], 2)
					);
					if (shapes[i_shape]->distance(cur_point) < epsilon)
					{
						is_valid(c[0], c[1], c[2]) = false;
						shapes[i_shape]->cluster.coords.emplace_back(c);
						shapes[i_shape]->cluster.surface_points.emplace_back(cur_point);
						shapes[i_shape]->inliers.emplace_back(cur_point);
						for (int dx = -1; dx <= 1; dx++)
							for (int dy = -1; dy <= 1; dy++)
								for (int dz = -1; dz <= 1; dz++)
								{
									if (dx == 0 && dy == 0 && dz == 0)
										continue;
									const int nx = c[0] + dx;
									const int ny = c[1] + dy;
									const int nz = c[2] + dz;
									if (!check_range(nx, ny, nz, res))
										continue;
									if (!is_valid(nx, ny, nz))
										continue;
									const Eigen::Vector3d p1(
										v_surface_points(c[0], c[1], c[2], 0),
										v_surface_points(c[0], c[1], c[2], 1),
										v_surface_points(c[0], c[1], c[2], 2));
									const Eigen::Vector3d p2(
										v_surface_points(nx, ny, nz, 0),
										v_surface_points(nx, ny, nz, 1),
										v_surface_points(nx, ny, nz, 2));
									is_valid(nx, ny, nz) = false;
									if ((p1 - p2).norm() < nearby_threshold)
										queues.emplace(nx, ny, nz);
								}
					}
				}
			}
		}
	);

	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (shapes[i_shape]->type != "curve")
					continue;

				std::shared_ptr<MyPlane> p = dynamic_pointer_cast<MyPlane>(
					fit_surface(shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, "plane").first);

				shapes[i_shape] = fit_curve(shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, p->plane, shapes[i_shape]->detail_type).first;
				shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
				shapes[i_shape]->find_boundary();
			}
		});

	return;
}

void boundary_grow_curve(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid,
	const Point_set& surface_boundary
)
{
	const double distance_to_boundary = 1e-3;
	const double nearby_threshold = 1e-2;
	const int res = v_surface_points.dimension(0);

	Eigen::Tensor<bool, 3> is_valid_global = v_is_valid;
	is_valid_global.setConstant(false);

	std::shared_ptr<my_kd_tree_t> kdtree(initialize_kd_tree(surface_boundary));
	tbb::parallel_for(tbb::blocked_range<int>(0, res * res * res), [&](const auto& r0)
		{
			for (int i = r0.begin(); i < r0.end(); ++i)
			{
				const int x = i / res / res;
				const int y = i / res % res;
				const int z = i % res;
				const Eigen::Vector3f p1(
					v_surface_points(x, y, z, 0),
					v_surface_points(x, y, z, 1),
					v_surface_points(x, y, z, 2));
				if (search_k_neighbour(*kdtree.get(), p1, 1).second[0] < distance_to_boundary * distance_to_boundary
					&& v_is_valid(x, y, z))
					is_valid_global(x, y, z) = true;
			}
		});
	
	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (shapes[i_shape]->type != "curve")
					continue;

				Eigen::Tensor<bool, 3> is_valid = is_valid_global;
				std::queue<Eigen::Vector3i> queues;
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					is_valid(c[0], c[1], c[2]) = false;
				}
				for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
				{
					const auto& c = shapes[i_shape]->cluster.coords[i_coord];
					for (int dx = -1; dx <= 1; dx++)
						for (int dy = -1; dy <= 1; dy++)
							for (int dz = -1; dz <= 1; dz++)
							{
								if (dx == 0 && dy == 0 && dz == 0)
									continue;
								const int nx = c[0] + dx;
								const int ny = c[1] + dy;
								const int nz = c[2] + dz;
								if (!check_range(nx, ny, nz, res))
									continue;
								if (!is_valid(nx, ny, nz))
									continue;
								is_valid(nx, ny, nz) = false;
								const Eigen::Vector3d p1(
									v_surface_points(c[0], c[1], c[2], 0),
									v_surface_points(c[0], c[1], c[2], 1),
									v_surface_points(c[0], c[1], c[2], 2));
								const Eigen::Vector3d p2(
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 0),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 1),
									v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 2));
								if ((p1 - p2).norm() < nearby_threshold)
									queues.emplace(c[0] + dx, c[1] + dy, c[2] + dz);
							}
				}
				while (!queues.empty())
				{
					const auto c = queues.front();
					queues.pop();

					Eigen::Vector3d cur_point(
						v_surface_points(c[0], c[1], c[2], 0),
						v_surface_points(c[0], c[1], c[2], 1),
						v_surface_points(c[0], c[1], c[2], 2)
					);
					if (shapes[i_shape]->distance(cur_point) < epsilon)
					{
						is_valid(c[0], c[1], c[2]) = false;
						shapes[i_shape]->cluster.coords.emplace_back(c);
						shapes[i_shape]->cluster.surface_points.emplace_back(cur_point);
						shapes[i_shape]->inliers.emplace_back(cur_point);
						for (int dx = -1; dx <= 1; dx++)
							for (int dy = -1; dy <= 1; dy++)
								for (int dz = -1; dz <= 1; dz++)
								{
									if (dx == 0 && dy == 0 && dz == 0)
										continue;
									const int nx = c[0] + dx;
									const int ny = c[1] + dy;
									const int nz = c[2] + dz;
									if (!check_range(nx, ny, nz, res))
										continue;
									if (!is_valid(nx, ny, nz))
										continue;
									const Eigen::Vector3d p1(
										v_surface_points(c[0], c[1], c[2], 0),
										v_surface_points(c[0], c[1], c[2], 1),
										v_surface_points(c[0], c[1], c[2], 2));
									const Eigen::Vector3d p2(
										v_surface_points(nx, ny, nz, 0),
										v_surface_points(nx, ny, nz, 1),
										v_surface_points(nx, ny, nz, 2));
									is_valid(nx, ny, nz) = false;
									if ((p1 - p2).norm() < nearby_threshold)
										queues.emplace(nx, ny, nz);
								}
					}
				}
			}
		}
	);

	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const auto& r0)
		{
			for (int i_shape = r0.begin(); i_shape < r0.end(); ++i_shape)
			{
				if (shapes[i_shape]->type != "curve")
					continue;

				std::shared_ptr<MyPlane> p = dynamic_pointer_cast<MyPlane>(
					fit_surface(shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, "plane").first);

				shapes[i_shape] = fit_curve(shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, p->plane, shapes[i_shape]->detail_type).first;
				shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
				shapes[i_shape]->find_boundary();
			}
		});

	return;
}

void boundary_grow_vertex(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<bool, 3>& v_is_valid
)
{
	const int res = v_surface_points.dimension(0);

	// For Surface
	#pragma omp parallel for schedule(dynamic)
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
	{
		if (shapes[i_shape]->type != "vertex")
			continue;

		Eigen::Tensor<bool, 3> is_valid = v_is_valid;
		std::queue<Eigen::Vector3i> queues;
		for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
		{
			const auto& c = shapes[i_shape]->cluster.coords[i_coord];
			is_valid(c[0], c[1], c[2]) = false;
		}
		for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
		{
			const auto& c = shapes[i_shape]->cluster.coords[i_coord];
			for (int dx = -1; dx <= 1; dx++)
				for (int dy = -1; dy <= 1; dy++)
					for (int dz = -1; dz <= 1; dz++)
					{
						if (dx == 0 && dy == 0 && dz == 0)
							continue;
						const int nx = c[0] + dx;
						const int ny = c[1] + dy;
						const int nz = c[2] + dz;
						if (!check_range(nx, ny, nz, res))
							continue;
						if (!is_valid(nx, ny, nz))
							continue;
						is_valid(nx, ny, nz) = false;
						const Eigen::Vector3d p1(
							v_surface_points(c[0], c[1], c[2], 0),
							v_surface_points(c[0], c[1], c[2], 1),
							v_surface_points(c[0], c[1], c[2], 2));
						const Eigen::Vector3d p2(
							v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 0),
							v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 1),
							v_surface_points(c[0] + dx, c[1] + dy, c[2] + dz, 2));
						if ((p1 - p2).norm() < 0.02)
							queues.emplace(c[0] + dx, c[1] + dy, c[2] + dz);
					}
		}
		while (!queues.empty())
		{
			const auto c = queues.front();
			queues.pop();

			Eigen::Vector3d cur_point(
				v_surface_points(c[0], c[1], c[2], 0),
				v_surface_points(c[0], c[1], c[2], 1),
				v_surface_points(c[0], c[1], c[2], 2)
			);
			if (shapes[i_shape]->distance(cur_point) < epsilon)
			{
				is_valid(c[0], c[1], c[2]) = false;
				shapes[i_shape]->cluster.coords.emplace_back(c);
				shapes[i_shape]->cluster.surface_points.emplace_back(cur_point);
				shapes[i_shape]->inliers.emplace_back(cur_point);
				for (int dx = -1; dx <= 1; dx++)
					for (int dy = -1; dy <= 1; dy++)
						for (int dz = -1; dz <= 1; dz++)
						{
							if (dx == 0 && dy == 0 && dz == 0)
								continue;
							const int nx = c[0] + dx;
							const int ny = c[1] + dy;
							const int nz = c[2] + dz;
							if (!check_range(nx, ny, nz, res))
								continue;
							if (!is_valid(nx, ny, nz))
								continue;
							const Eigen::Vector3d p1(
								v_surface_points(c[0], c[1], c[2], 0),
								v_surface_points(c[0], c[1], c[2], 1),
								v_surface_points(c[0], c[1], c[2], 2));
							const Eigen::Vector3d p2(
								v_surface_points(nx, ny, nz, 0),
								v_surface_points(nx, ny, nz, 1),
								v_surface_points(nx, ny, nz, 2));
							is_valid(nx, ny, nz) = false;
							if ((p1 - p2).norm() < 0.02)
								queues.emplace(nx, ny, nz);
						}
			}
		}
	}

	#pragma omp parallel for
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
	{
		if (shapes[i_shape]->type == "vertex")
		{
			shapes[i_shape] = fit_vertex(shapes[i_shape]->cluster.surface_points, shapes[i_shape]->cluster, epsilon).first;
			shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
			shapes[i_shape]->find_boundary();
		}
	}

	return;
}

void boundary_grow_restricted(
	const Eigen::Tensor<double, 4>& v_surface_points,
	std::vector<std::shared_ptr<Shape>>& shapes,
	const double epsilon,
	const Eigen::Tensor<double, 4>& features,
	const double udf_threshold
)
{
	const int resolution = v_surface_points.dimension(0);

	Eigen::Tensor<bool, 3> is_valid(resolution, resolution, resolution);
	is_valid.setConstant(true);
	#pragma omp parallel for
	for (int x = 0; x < resolution; ++x)
	{
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
				if (features(x, y, z, 0) > udf_threshold)
					is_valid(x, y, z) = false;
	}

	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
	{
		for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
		{
			const auto& coord = shapes[i_shape]->cluster.coords[i_coord];
			is_valid(
				coord[0],
				coord[1],
				coord[2]) = false;
		}
	}

	std::vector<std::string> orders{ "vertex", "curve", "surface" };
	for(const auto& type: orders)
	{
		for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		{
			if (shapes[i_shape]->type != type)
				continue;

			std::queue<Eigen::Vector3i> queues;
			for (int i_coord = 0; i_coord < shapes[i_shape]->cluster.coords.size(); ++i_coord)
			{
				const auto& coord = shapes[i_shape]->cluster.coords[i_coord];
				add_neighbours(
					coord[0],
					coord[1],
					coord[2],
					v_surface_points,
					queues, 0.02);
			}
			while (!queues.empty())
			{
				const auto cur_coor = queues.front();
				queues.pop();
				if (cur_coor[0] < 0 || cur_coor[0] >= resolution ||
					cur_coor[1] < 0 || cur_coor[1] >= resolution ||
					cur_coor[2] < 0 || cur_coor[2] >= resolution)
					continue;
				if (!is_valid(cur_coor[0], cur_coor[1], cur_coor[2]))
					continue;
				Eigen::Vector3d cur_point(
					v_surface_points(cur_coor[0], cur_coor[1], cur_coor[2], 0),
					v_surface_points(cur_coor[0], cur_coor[1], cur_coor[2], 1),
					v_surface_points(cur_coor[0], cur_coor[1], cur_coor[2], 2)
				);
				if (shapes[i_shape]->distance(cur_point) < epsilon)
				{
					is_valid(cur_coor[0], cur_coor[1], cur_coor[2]) = false;
					shapes[i_shape]->cluster.coords.push_back(cur_coor);
					shapes[i_shape]->cluster.surface_points.push_back(cur_point);
					shapes[i_shape]->cluster.query_points.push_back(cur_point);
					shapes[i_shape]->inliers.push_back(cur_point);
					add_neighbours(
						cur_coor[0], cur_coor[1], cur_coor[2], v_surface_points, queues, 0.02);
				}
			}
		}
	}

	#pragma omp parallel for
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
	{
		shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
		shapes[i_shape]->find_boundary();
	}

	#pragma omp parallel for
	for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
	{
		if (shapes[i_shape]->type == "vertex")
		{
			shapes[i_shape] = fit_vertex(shapes[i_shape]->inliers, shapes[i_shape]->cluster, epsilon).first;
			shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
			shapes[i_shape]->find_boundary();
		}
		if (shapes[i_shape]->type == "curve")
		{
			std::shared_ptr<MyPlane> p = dynamic_pointer_cast<MyPlane>(fit_surface(shapes[i_shape]->inliers, shapes[i_shape]->cluster, "plane").first);

			shapes[i_shape] = fit_curve(shapes[i_shape]->inliers, shapes[i_shape]->cluster, (p->plane), shapes[i_shape]->detail_type).first;
			shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
			shapes[i_shape]->find_boundary();
		}
		if (shapes[i_shape]->type == "surface")
		{
			shapes[i_shape] = fit_surface(shapes[i_shape]->inliers, shapes[i_shape]->cluster, shapes[i_shape]->detail_type).first;
			shapes[i_shape]->get_inliers(shapes[i_shape]->cluster.surface_points, epsilon);
			shapes[i_shape]->find_boundary();
		}
	}

	return;
}
// #pragma optimize ("", on)
