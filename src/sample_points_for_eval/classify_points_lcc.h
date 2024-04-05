#pragma once
#include <CGAL/Side_of_triangle_mesh.h>

#include "classify_points_region_growing.h"
#include "filling_holes.h"

std::pair<std::vector<Point_set>, std::vector<int>> build_point_clouds(const std::vector<CGAL::Polyhedron_3<K>>& polys)
{
	CGAL::Random_points_in_cube_3<Point_3> point_generator(1);
	std::vector<Point_3> sample_points;

	std::copy_n(point_generator, 1000000, std::back_inserter(sample_points));

	std::vector<CGAL::Side_of_triangle_mesh<CGAL::Polyhedron_3<K>, K>> insides;
	for (int i_poly = 0; i_poly < polys.size(); ++i_poly)
		insides.emplace_back(polys[i_poly]);

	std::vector<int> flags(sample_points.size(), -1);

#pragma omp parallel for
	for (int i = 0; i < sample_points.size(); ++i)
	{
		for (int i_poly = 0; i_poly < polys.size(); ++i_poly)
		{
			if (insides[i_poly](sample_points[i]) == CGAL::ON_BOUNDED_SIDE)
			{
				flags[i] = i_poly;
				break;
			}
		}
	}

	std::vector<Point_set> points(polys.size());
	Point_set nowhere;
	for (int i = 0; i < flags.size(); ++i)
	{
		if (flags[i] == -1)
			nowhere.insert(sample_points[i]);
		else
			points[flags[i]].insert(sample_points[i]);
	}
	for (int i_poly = 0; i_poly < polys.size(); ++i_poly)
		CGAL::IO::write_point_set((boost::format("temp/cell/point_%d.ply") % i_poly).str(), points[i_poly]);
	CGAL::IO::write_point_set((boost::format("temp/cell/00.ply")).str(), nowhere);

	return std::make_pair(points, flags);
}

// Trilinear interpolation
Eigen::Vector4d get_trilinear_feature(
	const Eigen::Tensor<double, 4>& v_gradients,
	const Eigen::Tensor<double, 3>& v_udf,
	const Eigen::Vector3d& v_coord,
	const int v_resolution)
{
	const int x0 = std::floor(v_coord[0]);
	const int y0 = std::floor(v_coord[1]);
	const int z0 = std::floor(v_coord[2]);
	const int x1 = std::ceil(v_coord[0]);
	const int y1 = std::ceil(v_coord[1]);
	const int z1 = std::ceil(v_coord[2]);

	const double xd = (v_coord[0] - x0) / (x1 - x0);
	const double yd = (v_coord[1] - y0) / (y1 - y0);
	const double zd = (v_coord[2] - z0) / (z1 - z0);

	const double udf00 = v_udf(x0, y0, z0) * (1 - xd) + v_udf(x1, y0, z0) * xd;
	const double udf01 = v_udf(x0, y0, z1) * (1 - xd) + v_udf(x1, y0, z1) * xd;
	const double udf10 = v_udf(x0, y1, z0) * (1 - xd) + v_udf(x1, y1, z0) * xd;
	const double udf11 = v_udf(x0, y1, z1) * (1 - xd) + v_udf(x1, y1, z1) * xd;

	const double udf0 = udf00 * (1 - yd) + udf10 * yd;
	const double udf1 = udf01 * (1 - yd) + udf11 * yd;
	const double udf = udf0 * (1 - zd) + udf1 * zd;

	Eigen::Vector3d g000 = get_vector(v_gradients, x0, y0, z0);
	Eigen::Vector3d g001 = get_vector(v_gradients, x0, y0, z1);
	Eigen::Vector3d g010 = get_vector(v_gradients, x0, y1, z0);
	Eigen::Vector3d g011 = get_vector(v_gradients, x0, y1, z1);
	Eigen::Vector3d g100 = get_vector(v_gradients, x1, y0, z0);
	Eigen::Vector3d g101 = get_vector(v_gradients, x1, y0, z1);
	Eigen::Vector3d g110 = get_vector(v_gradients, x1, y1, z0);
	Eigen::Vector3d g111 = get_vector(v_gradients, x1, y1, z1);

	const double phi000 = std::atan2(g000[1], g000[0]);
	const double theta000 = std::asin(g000[2]);
	const double phi001 = std::atan2(g001[1], g001[0]);
	const double theta001 = std::asin(g001[2]);
	const double phi010 = std::atan2(g010[1], g010[0]);
	const double theta010 = std::asin(g010[2]);
	const double phi011 = std::atan2(g011[1], g011[0]);
	const double theta011 = std::asin(g011[2]);
	const double phi100 = std::atan2(g100[1], g100[0]);
	const double theta100 = std::asin(g100[2]);
	const double phi101 = std::atan2(g101[1], g101[0]);
	const double theta101 = std::asin(g101[2]);
	const double phi110 = std::atan2(g110[1], g110[0]);
	const double theta110 = std::asin(g110[2]);
	const double phi111 = std::atan2(g111[1], g111[0]);
	const double theta111 = std::asin(g111[2]);

	const double phi00 = phi000 * (1 - xd) + phi100 * xd;
	const double phi01 = phi001 * (1 - xd) + phi101 * xd;
	const double phi10 = phi010 * (1 - xd) + phi110 * xd;
	const double phi11 = phi011 * (1 - xd) + phi111 * xd;

	const double phi0 = phi00 * (1 - yd) + phi10 * yd;
	const double phi1 = phi01 * (1 - yd) + phi11 * yd;
	const double phi = phi0 * (1 - zd) + phi1 * zd;

	const double theta00 = theta000 * (1 - xd) + theta100 * xd;
	const double theta01 = theta001 * (1 - xd) + theta101 * xd;
	const double theta10 = theta010 * (1 - xd) + theta110 * xd;
	const double theta11 = theta011 * (1 - xd) + theta111 * xd;

	const double theta0 = theta00 * (1 - yd) + theta10 * yd;
	const double theta1 = theta01 * (1 - yd) + theta11 * yd;
	const double theta = theta0 * (1 - zd) + theta1 * zd;

	const double dx = std::cos(theta) * std::cos(phi);
	const double dy = std::cos(theta) * std::sin(phi);
	const double dz = std::sin(theta);

	return Eigen::Vector4d(udf, dx, dy, dz);
}


std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector4d>>> get_interpolated_feature(
	const std::vector<Point_set>& v_points,
	const Eigen::Tensor<double, 4>& v_gradients,
	const Eigen::Tensor<double, 3>& v_udf,
	const int v_resolution)
{
	std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector4d>>> results(v_points.size());
	for (int i_set = 0; i_set < v_points.size(); ++i_set)
	{
		const Point_set& point_set = v_points[i_set];
		results[i_set].resize(point_set.size());
#pragma omp parallel for
		for (int i_point = 0; i_point < point_set.size(); ++i_point)
		{
			Eigen::Vector3d coord = ((cgal_2_eigen_point<double>(point_set.point(i_point)) -
				Eigen::Vector3d(-1, -1, -1)) / 2.0 * (v_resolution - 1));
			Eigen::Vector4d pos = get_trilinear_feature(v_gradients, v_udf, coord, v_resolution);
			results[i_set][i_point] = std::make_pair(cgal_2_eigen_point<double>(point_set.point(i_point)), pos);
		}
	}
	return results;
}

void classify_points_lcc(const std::vector<Point_3>& alpha_points,
                         const std::vector<std::vector<unsigned long long>>& alpha_polygons,
                         bool store_cells, int resolution,
                         const Eigen::Tensor<double, 4>& gradients,
                         const Eigen::Tensor<bool, 3>& consistent_flags,
                         const Eigen::Tensor<double, 3>& udf
)
{
	LCC_3 lcc;
	std::vector<CGAL::Polyhedron_3<K>> polys;
	build_lcc(alpha_points, alpha_polygons, lcc, polys, store_cells);
	// Visualize
	{
		Point_set dilated_boundary_points;
		for (int x = 0; x < resolution; ++x)
			for (int y = 0; y < resolution; ++y)
				for (int z = 0; z < resolution; ++z)
				{
					if (!consistent_flags(x, y, z))
						continue;

					Eigen::Vector3d cur_pos(x, y, z);
					cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();

					dilated_boundary_points.insert(eigen_2_cgal_point(cur_pos));
				}
		CGAL::IO::write_point_set("temp/summary/hole_filling_boundary.ply", dilated_boundary_points);
	}

	// Dilate
	if (false)
	{
		Eigen::Tensor<bool, 3> dilated_flags = consistent_flags;
		{
			// const int half_window_size = program.get<int>("--threshold");
			const int half_window_size = 0;
			throw;
			for (int x = 0; x < resolution; ++x)
				for (int y = 0; y < resolution; ++y)
					for (int z = 0; z < resolution; ++z)
					{
						if (x < half_window_size || x >= resolution - half_window_size ||
							y < half_window_size || y >= resolution - half_window_size ||
							z < half_window_size || z >= resolution - half_window_size)
						{
							dilated_flags(x, y, z) = true;
							continue;
						}

						bool flag = consistent_flags(x, y, z);
						for (int dx = -1; dx <= half_window_size; ++dx)
							for (int dy = -1; dy <= half_window_size; ++dy)
								for (int dz = -1; dz <= half_window_size; ++dz)
									if (consistent_flags(x + dx, y + dy, z + dz))
										flag = true;
						dilated_flags(x, y, z) = flag;
					}
			// Visualize
			{
				Point_set dilated_boundary_points;
				for (int x = 0; x < resolution; ++x)
					for (int y = 0; y < resolution; ++y)
						for (int z = 0; z < resolution; ++z)
						{
							if (!dilated_flags(x, y, z))
								continue;

							Eigen::Vector3d cur_pos(x, y, z);
							cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();

							dilated_boundary_points.insert(eigen_2_cgal_point(cur_pos));
						}
				CGAL::IO::write_point_set("temp/total_dilated_boundary.ply", dilated_boundary_points);
			}
		}

		//consistent_flags = dilated_flags;
	}

	auto results = build_point_clouds(polys);
	auto interpolated_results = get_interpolated_feature(results.first, gradients, udf, resolution);

	// Visualize
	{
		Point_set total_points;
		auto r_table = total_points.add_property_map<uchar>("red", 0).first;
		auto g_table = total_points.add_property_map<uchar>("green", 0).first;
		auto b_table = total_points.add_property_map<uchar>("blue", 0).first;
		auto color_table = get_color_table_bgr2();
		for (int i_cluster = 0; i_cluster < interpolated_results.size(); ++i_cluster)
		{
			Point_set cluster_points;
			for (const auto& point : interpolated_results[i_cluster])
			{
				Eigen::Vector3d pos = point.first;
				Eigen::Vector3d dir(point.second[1], point.second[2], point.second[3]);
				double udf = point.second[0];
				dir.normalize();

				Eigen::Vector3d p = pos + dir * udf;

				cluster_points.insert(eigen_2_cgal_point(p));
				total_points.insert(eigen_2_cgal_point(p));
				r_table[total_points.size() - 1] = color_table[i_cluster % color_table.size()][2];
				g_table[total_points.size() - 1] = color_table[i_cluster % color_table.size()][1];
				b_table[total_points.size() - 1] = color_table[i_cluster % color_table.size()][0];
			}
			CGAL::IO::write_point_set((ffmt("temp/cluster/%d.ply") % i_cluster).str(), cluster_points);
		}
		CGAL::IO::write_point_set("temp/summary/total_surface.ply", total_points);
	}
}
