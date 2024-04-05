#include "tools.h"

void add_neighbours(const int x, const int y, const int z, std::queue<std::tuple<int, int, int>>& v_queues)
{
	v_queues.push({ x - 1, y - 1, z - 1 });
	v_queues.push({ x - 1, y - 1, z });
	v_queues.push({ x - 1, y - 1, z + 1 });

	v_queues.push({ x - 1, y, z - 1 });
	v_queues.push({ x - 1, y, z });
	v_queues.push({ x - 1, y, z + 1 });

	v_queues.push({ x - 1, y + 1, z - 1 });
	v_queues.push({ x - 1, y + 1, z });
	v_queues.push({ x - 1, y + 1, z + 1 });

	v_queues.push({ x, y - 1, z - 1 });
	v_queues.push({ x, y - 1, z });
	v_queues.push({ x, y - 1, z + 1 });

	v_queues.push({ x, y, z - 1 });
	v_queues.push({ x, y, z + 1 });

	v_queues.push({ x, y + 1, z - 1 });
	v_queues.push({ x, y + 1, z });
	v_queues.push({ x, y + 1, z + 1 });

	v_queues.push({ x + 1, y - 1, z - 1 });
	v_queues.push({ x + 1, y - 1, z });
	v_queues.push({ x + 1, y - 1, z + 1 });

	v_queues.push({ x + 1, y, z - 1 });
	v_queues.push({ x + 1, y, z });
	v_queues.push({ x + 1, y, z + 1 });

	v_queues.push({ x + 1, y + 1, z - 1 });
	v_queues.push({ x + 1, y + 1, z });
	v_queues.push({ x + 1, y + 1, z + 1 });

	return;
}

void add_neighbours(const int x, const int y, const int z, const Eigen::Tensor<double, 4>& v_surface_points,
	std::queue<Eigen::Vector3i>& v_queues, const double v_epsilon)
{
	const int res = static_cast<int>(v_surface_points.dimension(0));
	for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dz = -1; dz <= 1; dz++)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;
				if (x + dx < 0 || x + dx >= res ||
					y + dy < 0 || y + dy >= res ||
					z + dz < 0 || z + dz >= res)
					continue;
				const Eigen::Vector3d p1(
					v_surface_points(x, y, z, 0),
					v_surface_points(x, y, z, 1),
					v_surface_points(x, y, z, 2));
				const Eigen::Vector3d p2(
					v_surface_points(x + dx, y + dy, z + dz, 0),
					v_surface_points(x + dx, y + dy, z + dz, 1),
					v_surface_points(x + dx, y + dy, z + dz, 2));
				if ((p1 - p2).norm() < v_epsilon)
					v_queues.emplace(x + dx, y + dy, z + dz);
			}
	return;
}

bool check_range(const int x, const int y, const int z, const int resolution)
{
	return x >= 0 && x < resolution &&
		y >= 0 && y < resolution &&
		z >= 0 && z < resolution;
}

Eigen::Vector3d get_vector(const Eigen::Tensor<double, 4>& v_tensor, const long long x, const long long y,
	const long long z)
{
	const auto& dimension = v_tensor.dimensions();
	const long long base_index = z * dimension[0] * dimension[1] * dimension[2] + y * dimension[0] * dimension[1] + x *
		dimension[0];
	const double* ptr = v_tensor.data() + base_index;
	return Eigen::Vector3d{ *ptr, *(ptr + 1), *(ptr + 2) };
}

void export_points(const std::string& v_path, const Eigen::Tensor<bool, 3>& consistent_flags, const int resolution)
{
	Point_set boundary_points;
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
			{
				if (!consistent_flags(x, y, z))
					continue;

				Eigen::Vector3d cur_pos(x, y, z);
				cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();

				boundary_points.insert(eigen_2_cgal_point(cur_pos));
			}
	CGAL::IO::write_point_set(v_path, boundary_points);
}

void export_points(const std::string& v_path, const Eigen::Tensor<bool, 3>& consistent_flags, const int resolution,
	const Eigen::Tensor<double, 4>& features, const double threshold)
{
	Point_set boundary_points;
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
			{
				if (!consistent_flags(x, y, z))
					continue;

				if (features(x, y, z, 0) > threshold)
					continue;

				Eigen::Vector3d cur_pos(x, y, z);
				cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();

				boundary_points.insert(eigen_2_cgal_point(cur_pos));
			}
	CGAL::IO::write_point_set(v_path, boundary_points);
}

