#include "filling_holes.h"


#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <CGAL/Linear_cell_complex_for_combinatorial_map.h>
#include <CGAL/Linear_cell_complex_incremental_builder_3.h>
#include <CGAL/draw_linear_cell_complex.h>

#include <tbb/tbb.h>

#include "tools.h"
#include "model_tools.h"

typedef CGAL::Fixed_alpha_shape_vertex_base_3<K> F_Vb;
typedef CGAL::Fixed_alpha_shape_cell_base_3<K> F_Fb;
typedef CGAL::Triangulation_data_structure_3<F_Vb, F_Fb> F_Tds;
typedef CGAL::Delaunay_triangulation_3<K, F_Tds> F_Delaunay;
typedef CGAL::Fixed_alpha_shape_3<F_Delaunay> Fixed_alpha_shape_3;
typedef F_Delaunay::Cell_handle F_Cell_handle;

typedef CGAL::Alpha_shape_vertex_base_3<K> Vb;
typedef CGAL::Alpha_shape_cell_base_3<K> Fb;
typedef CGAL::Triangulation_data_structure_3<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;
typedef CGAL::Alpha_shape_3<Delaunay> Alpha_shape_3;
typedef Delaunay::Cell_handle Cell_handle;
typedef Alpha_shape_3::Facet Facet;

using Point = LCC_3::Point;


/*
 * Utils
 */

std::pair<Eigen::Tensor<bool, 3>, Eigen::VectorXi> mark_boundary(
	const Eigen::Tensor<bool, 3>& v_flags, 
	const Eigen::Tensor<double, 4>& v_udf,
	const double v_distance_threshold
)
{
	Eigen::Tensor<bool, 3> new_flags(v_flags);
	const int boundary = 2;
	auto size = v_flags.dimensions();
	int min_x = size[0];
	int min_y = size[1];
	int min_z = size[2];
	int max_x = 0;
	int max_y = 0;
	int max_z = 0;

	for (int x = 0; x < size[0]; x += 1)
		for (int y = 0; y < size[1]; y += 1)
			for (int z = 0; z < size[2]; z += 1)
			{
				if (x < boundary || x >= size[0] - boundary || y < boundary || y >= size[1] - boundary || z < boundary
					|| z >= size[2] - boundary)
					new_flags(x, y, z) = true;

				if (v_udf(x, y, z, 0) > v_distance_threshold)
				{
					new_flags(x, y, z) = true;
				}
				else
				{
					min_x = x > min_x ? min_x : x;
					min_y = y > min_y ? min_y : y;
					min_z = z > min_z ? min_z : z;
					max_x = x < max_x ? max_x : x;
					max_y = y < max_y ? max_y : y;
					max_z = z < max_z ? max_z : z;
				}
				// if (
				// 	(x < boundary || x >= size[0] - boundary) && (y < boundary || y >= size[1] - boundary) ||
				// 	(x < boundary || x >= size[0] - boundary) && (z < boundary || z >= size[2] - boundary) ||
				// 	(y < boundary || y >= size[1] - boundary) && (z < boundary || z >= size[2] - boundary)
				// 	)
				// 	new_flags(x, y, z) = false;
			}
	Eigen::VectorXi bounds(6);
	bounds << min_x, min_y, min_z, max_x, max_y, max_z;
	return std::make_pair(new_flags, bounds);
}

Eigen::Tensor<bool, 3> voxelize(const int resolution, const std::vector<Triangle_3>& v_triangles)
{
	LOG(INFO) << "Start to voxelize the alpha shape";
	Eigen::Tensor<bool, 3> new_flags(resolution, resolution, resolution);
	new_flags.setConstant(false);
	Point_set sampled_points = sample_points_according_density(v_triangles, resolution * resolution * 10);

	for (int i = 0; i < sampled_points.size(); ++i)
	{
		int x = (sampled_points.point(i).x() - (-1)) / 2 * resolution;
		int y = (sampled_points.point(i).y() - (-1)) / 2 * resolution;
		int z = (sampled_points.point(i).z() - (-1)) / 2 * resolution;
		if (x <= 0 || x > resolution - 1 || y <= 0 || y > resolution - 1 || z <= 0 || z > resolution - 1)
			continue;
		new_flags(x, y, z) = true;
	}

	CGAL::IO::write_point_set("temp/sampled_alpha_shapes.ply", sampled_points);
	return new_flags;
}

Eigen::Tensor<bool, 3> dilate(const Eigen::Tensor<bool, 3>& v_flag, const int resolution, const int window_size)
{

	// Dilate
	Eigen::Tensor<bool, 3> dilated_flags = v_flag;
	if (window_size <= 0)
		return dilated_flags;
	{
		for (int x = 0; x < resolution; ++x)
			for (int y = 0; y < resolution; ++y)
				for (int z = 0; z < resolution; ++z)
				{
					if (x > resolution - window_size ||
						y > resolution - window_size ||
						z > resolution - window_size)
					{
						// dilated_flags(x, y, z) = true;
						continue;
					}

					bool flag = v_flag(x, y, z);
					for (int dx = 0; dx < window_size; ++dx)
						for (int dy = 0; dy < window_size; ++dy)
							for (int dz = 0; dz < window_size; ++dz)
								if (v_flag(x + dx, y + dy, z + dz))
									flag = true;
					dilated_flags(x, y, z) = flag;
				}
	}

	return dilated_flags;
}

/*
 * Alpha shape related
 */

Eigen::Tensor<bool, 3> rebuild_flags(const std::vector<Triangle_3>& v_triangles,
	const Eigen::Tensor<bool, 3>& v_edge_flag, const int v_num_per_m2)
{
	Eigen::Tensor<bool, 3> o_edge_flag = v_edge_flag;

	Point_set sampled_points = sample_points_according_density(v_triangles, v_num_per_m2);
	for (int i_point = 0; i_point < sampled_points.size(); ++i_point)
	{
		const int x = std::round((sampled_points.point(i_point).x() + 1) / 2 * 255);
		const int y = std::round((sampled_points.point(i_point).y() + 1) / 2 * 255);
		const int z = std::round((sampled_points.point(i_point).z() + 1) / 2 * 255);
		o_edge_flag(x, y, z) = true;
	}
	return o_edge_flag;
}

// Return the points and the polygons of the alpha shapes
// Will write ply file "alpha_shapes_soup.ply"
std::vector<Triangle_3> filling_holes(
	const Point_set& boundary_points, const double alpha_value)
{
	LOG(INFO) << "Start to calculate alpha shapes using alpha values^2=" << alpha_value;
	Alpha_shape_3 as(boundary_points.points().begin(), boundary_points.points().end(),
	                 // alpha_value, Alpha_shape_3::REGULARIZED);
	                 alpha_value, Alpha_shape_3::GENERAL);

	std::vector<Alpha_shape_3::Facet> facets;
	as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::REGULAR);
	as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::SINGULAR);
	std::size_t nbf = facets.size();

	LOG(INFO) << boost::format("Calculate done; %d facets and %d cells") % nbf % as.number_of_cells();
	LOG(INFO) << "Start to extract surfaces";

	std::vector<Point_3> points;
	std::map<Alpha_shape_3::Vertex_handle, int> vertices;
	for (auto v_it = as.vertices_begin(); v_it != as.vertices_end(); v_it++)
	{
		vertices[v_it] = points.size();
		points.emplace_back(v_it->point());
	}

	std::vector<Triangle_3> triangles(facets.size());
	std::vector<std::vector<unsigned long long>> polygons(facets.size());

	#pragma omp parallel for
	for (int i = 0; i < facets.size(); ++i)
	//for (auto v_it : facets)
	{
		Alpha_shape_3::Facet cur_face = facets[i];
		if (as.classify(cur_face.first) != Alpha_shape_3::EXTERIOR)
			cur_face = as.mirror_facet(cur_face);
		if (as.classify(cur_face.first) != Alpha_shape_3::EXTERIOR)
		{
			std::cout << as.classify(cur_face);
			throw;
		}

		std::vector<unsigned long long> handles;
		handles.emplace_back(vertices.at(cur_face.first->vertex((cur_face.second + 1) % 4)));
		handles.emplace_back(vertices.at(cur_face.first->vertex((cur_face.second + 2) % 4)));
		handles.emplace_back(vertices.at(cur_face.first->vertex((cur_face.second + 3) % 4)));

		if (cur_face.second % 2 == 0) std::swap(handles[0], handles[1]);

		polygons[i] = handles;
		triangles[i] = Triangle_3(
			points[handles[0]],
			points[handles[1]],
			points[handles[2]]
		);
	}

	LOG(INFO) << "Found alpha value: " << as.get_alpha();
	write_ply("temp/alpha/alpha_shapes_triangles.ply", triangles);

	// LOG(INFO) << "Fix the orientation";
	// CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
	// CGAL::IO::write_PLY("temp/alpha/alpha_shapes_soup.ply", points, polygons);
	LOG(INFO) << "Done";

	if(false)
	{
		std::queue<Alpha_shape_3::Cell_handle> q;
		std::unordered_set<Alpha_shape_3::Cell_handle> visited_cells;

		int id_polyhedron = 0;
		for(auto it = as.alpha_shape_cells_begin(); it!= as.alpha_shape_cells_end(); ++it)
		{
			std::vector<Point_3> local_points;
			std::vector<std::vector<int>> local_polygons;
			std::map<Point_3, int> face_map;

			q.push(it.base());
			Cell_handle cur_cell;

			Point_set debug_set;
			while (!q.empty())
			{
				cur_cell = q.front();
				q.pop();

				if (visited_cells.find(cur_cell) != visited_cells.end())
					continue;
				visited_cells.insert(cur_cell);


				int id_verts[4];
				for (int i = 0; i < 4; ++i)
				{
					Point_3& p = cur_cell->vertex(i)->point();
					debug_set.insert(p);
					if (face_map.find(p) == face_map.end())
					{
						face_map.insert({ p, local_points.size() });
						local_points.push_back(p);
					}
					id_verts[i] = face_map.at(p);
				}
				if (as.classify(Facet(cur_cell, 0)) == Alpha_shape_3::INTERIOR || as.classify(Facet(cur_cell, 0)) == Alpha_shape_3::EXTERIOR)
					q.push(cur_cell->neighbor(0));
				else
					local_polygons.emplace_back(std::vector<int>{ id_verts[1], id_verts[2], id_verts[3] });
				if (as.classify(Facet(cur_cell, 1)) == Alpha_shape_3::INTERIOR || as.classify(Facet(cur_cell, 1)) == Alpha_shape_3::EXTERIOR)
					q.push(cur_cell->neighbor(1));
				else
					local_polygons.emplace_back(std::vector<int>{ id_verts[0], id_verts[2], id_verts[3] });
				if (as.classify(Facet(cur_cell, 2)) == Alpha_shape_3::INTERIOR || as.classify(Facet(cur_cell, 2)) == Alpha_shape_3::EXTERIOR)
					q.push(cur_cell->neighbor(2));
				else
					local_polygons.emplace_back(std::vector<int>{ id_verts[0], id_verts[1], id_verts[3] });
				if (as.classify(Facet(cur_cell, 3)) == Alpha_shape_3::INTERIOR || as.classify(Facet(cur_cell, 3)) == Alpha_shape_3::EXTERIOR)
					q.push(cur_cell->neighbor(3));
				else
					local_polygons.emplace_back(std::vector<int>{ id_verts[0], id_verts[1], id_verts[2] });
			}
			if(debug_set.empty())
				continue;
			CGAL::IO::write_point_set("temp/cell/p_"+std::to_string(id_polyhedron)+".ply", debug_set);

			Polyhedron_3 poly;
			CGAL::Polygon_mesh_processing::orient_polygon_soup(local_points, local_polygons);
			CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(local_points, local_polygons, poly);
			CGAL::IO::write_PLY("temp/cell/" + std::to_string(id_polyhedron) + ".ply", poly);
			id_polyhedron++;
		}
	}

	// Surface_mesh mesh;
	// CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
	// CGAL::IO::write_PLY("temp/alpha_shapes_mesh.ply", mesh);
	return triangles;
}


void build_lcc(const std::vector<Point_3>& v_points,
               const std::vector<std::vector<unsigned long long>>& v_faces,
               LCC_3& lcc,
               std::vector<CGAL::Polyhedron_3<K>>& polys,
			   bool is_viz_cells
)
{
	LOG(INFO) << "Start to build LCC";
	CGAL::Linear_cell_complex_incremental_builder_3<LCC_3> ib(lcc);
	for (const auto& v_it : v_points)
		ib.add_vertex(v_it);

	ib.begin_surface();
	for (const auto& face : v_faces)
	{
		ib.add_facet({face[0], face[1], face[2]});
	}
	ib.end_surface();
	// std::cout << lcc.are_all_faces_closed() << std::endl;
	// std::cout << CGAL::is_closed(lcc);
	lcc.display_characteristics(LOG(INFO));

	LOG(INFO) << "Start to extract polyhedrons";
	std::vector<LCC_3::Dart_handle> darts;
	for (auto it_cell = lcc.one_dart_per_cell<3>().begin(); it_cell != lcc.one_dart_per_cell<3>().end(); ++it_cell)
		darts.emplace_back(it_cell);

	polys.resize(darts.size());
	// #pragma omp parallel for
	for (int i=0;i<darts.size();++i)
	{
		const auto& it_cell = darts[i];
		std::vector<Point_3> local_points;
		std::map<Point_3, int> face_map;
		// std::map<LCC_3::Dart_handle, int> face_map;
		for (auto it_vertice = lcc.one_dart_per_incident_cell<0, 3>(it_cell).begin(); it_vertice != lcc.
		     one_dart_per_incident_cell<0, 3>(it_cell).end(); ++it_vertice)
		{
			// face_map.insert({ it_vertice, local_points.size() });
			face_map.insert({lcc.point(it_vertice), local_points.size()});
			local_points.push_back(lcc.point(it_vertice));
		}

		std::vector<std::vector<int>> local_polygons;
		for (auto it_face = lcc.one_dart_per_incident_cell<2, 3>(it_cell).begin(); it_face != lcc.
		     one_dart_per_incident_cell<2, 3>(it_cell).end(); ++it_face)
		{
			std::vector<int> local_polygons_;
			for (auto it_vertice = lcc.one_dart_per_incident_cell<0, 2>(it_face).begin(); it_vertice != lcc.
			     one_dart_per_incident_cell<0, 2>(it_face).end(); ++it_vertice)
			{
				local_polygons_.push_back(face_map.at(lcc.point(it_vertice)));
			}
			local_polygons.push_back(local_polygons_);
		}

		auto bbox = get_bounding_box(local_points);
		if ((bbox.min() - Eigen::Vector3f(-1.f, -1.f, -1.f)).norm() < 1e-3 &&
			(bbox.max() - Eigen::Vector3f(1.f, 1.f, 1.f)).norm() < 1e-3)
			continue;

		CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(local_points, local_polygons, polys[i]);
		if (is_viz_cells)
			CGAL::IO::write_PLY("temp/cell/mesh_" + std::to_string(polys.size()) + ".ply", local_points, local_polygons);

		// std::cout << P.is_closed() << P.is_valid() << std::endl;;
		// std::cout << CGAL::Polygon_mesh_processing::is_polygon_soup_a_polygon_mesh(local_points, local_polygons);
		// CGAL::Polygon_mesh_processing::orient_polygon_soup(local_points, local_polygons);
		// CGAL::IO::write_PLY("debug_cell1.ply", local_points, local_polygons);
		// CGAL::IO::write_PLY("debug_cell.ply", P);
	}

	return;
}

Eigen::Tensor<bool, 3> filling_hole(const Eigen::Tensor<bool, 3>& v_flag, const Eigen::Tensor<double, 4>& v_gradient, const int resolution)
{
	// Dilate
	Eigen::Tensor<bool, 3> dilated_flags = v_flag;

	const int window_size = 16;
	const double distance_threshold = 1;

	#pragma omp parallel for
	for (int x = 0; x < resolution - window_size; x += window_size)
		for (int y = 0; y < resolution - window_size; y += window_size)
			for (int z = 0; z < resolution - window_size; z += window_size)
			{
				std::vector<Eigen::Vector3d> poses;
				for (int dx = 0; dx < window_size; ++dx)
					for (int dy = 0; dy < window_size; ++dy)
						for (int dz = 0; dz < window_size; ++dz)
							if (v_flag(x + dx, y + dy, z + dz))
								poses.emplace_back(dx, dy, dz);
				if (poses.size() < window_size * window_size / 2)
					continue;

				Eigen::MatrixXd coord(3, poses.size());
				for (int i = 0; i < poses.size(); ++i)
					coord.col(i) = poses[i];

				Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());
				coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

				auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
				Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>().normalized();

				double d = -plane_normal.dot(centroid);

				// Calculate the orthogonal distance from the plane to the point
				for (int dx = 0; dx < window_size; ++dx)
					for (int dy = 0; dy < window_size; ++dy)
						for (int dz = 0; dz < window_size; ++dz)
							if (!v_flag(x + dx, y + dy, z + dz))
							{
								double dist = std::abs(plane_normal[0] * dx + plane_normal[1] * dy + plane_normal[2] * dz + d);
								if (dist < distance_threshold)
									dilated_flags(x + dx, y + dy, z + dz) = true;
							}
			}

	#pragma omp parallel for
	for (int x = window_size / 2; x < resolution - window_size; x += window_size)
		for (int y = window_size / 2; y < resolution - window_size; y += window_size)
			for (int z = window_size / 2; z < resolution - window_size; z += window_size)
			{
				std::vector<Eigen::Vector3d> poses;
				for (int dx = 0; dx < window_size; ++dx)
					for (int dy = 0; dy < window_size; ++dy)
						for (int dz = 0; dz < window_size; ++dz)
							if (v_flag(x + dx, y + dy, z + dz))
								poses.emplace_back(dx, dy, dz);
				if (poses.size() < window_size * window_size / 2)
					continue;

				Eigen::MatrixXd coord(3, poses.size());
				for (int i = 0; i < poses.size(); ++i)
					coord.col(i) = poses[i];

				Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());
				coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

				auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
				Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>().normalized();

				double d = -plane_normal.dot(centroid);

				// Calculate the orthogonal distance from the plane to the point
				for (int dx = 0; dx < window_size; ++dx)
					for (int dy = 0; dy < window_size; ++dy)
						for (int dz = 0; dz < window_size; ++dz)
							if (!v_flag(x + dx, y + dy, z + dz))
							{
								double dist = std::abs(plane_normal[0] * dx + plane_normal[1] * dy + plane_normal[2] * dz + d);
								if (dist < distance_threshold)
									dilated_flags(x + dx, y + dy, z + dz) = true;
							}
			}

	return dilated_flags;
}

/*
 * Alpha shape related
 */
Eigen::Tensor<bool, 3> dilate_according_to_gradients(
	const Eigen::Tensor<bool, 3>& v_flag,
	const Eigen::Tensor<double, 4>& v_features,
	const int window_size, const double v_udf_threshold
)
{
	Eigen::Tensor<bool, 3> new_flags = v_flag;
	const int res = new_flags.dimension(0);

	const double step_size = 1e-2;
	for (int i = 0; i < res * res * res; ++i)
	{
		const int x = i / res / res;
		const int y = i / res % res;
		const int z = i % res;
		if (!v_flag(x, y, z))
			continue;

		if (v_features(x, y, z, 0) > v_udf_threshold)
			continue;

		const double udf = v_features(x, y, z, 0);
		Eigen::Vector3d gradient(
			v_features(x, y, z, 1),
			v_features(x, y, z, 2),
			v_features(x, y, z, 3)
		);
		gradient.normalize();

		Eigen::Vector3d pos(x, y, z);
		pos = pos.array() / res * 2 - 1;
		for (int i_w = -window_size; i_w <= window_size; ++i_w)
		{
			Eigen::Vector3d new_p = pos + i_w * gradient * step_size;
			Eigen::Vector3i v_p = Eigen::round((new_p.array() + 1) / 2 * res).cast<int>();
			if (v_p.minCoeff() < 0 || v_p.maxCoeff() >= res)
				continue;
			new_flags(v_p(0), v_p(1), v_p(2)) = true;
		}
	}

	return new_flags;
}

Eigen::Tensor<bool, 3> dilate_along_the_ray(
	const Eigen::Tensor<double, 4>& v_features,
	const double v_udf_threshold,
	const Eigen::Tensor<bool, 3>& v_flags,
	const double threshold
)
{
	Eigen::Tensor<bool, 3> new_flags = v_flags;
	const int res = static_cast<int>(v_features.dimension(0));

	double ss = 2. / 256; // step size

	tbb::parallel_for(tbb::blocked_range<int>(0, res * res * res), [&](const auto& r0)
		{
			for (int i_point = r0.begin(); i_point != r0.end(); ++i_point)
			// for (int i_point = 0; i_point < res*res*res; ++i_point)
			{
				// i_point = (int)((0.0687949 + 1) / (2. / res)) * res * res + (int)((-0.174331 + 1) / (2. / res)) * res + (int)((-0.388449 + 1) / (2. / res));
				Eigen::Vector3i v_p(
					i_point / res / res,
					i_point / res % res,
					i_point % res
				);

				if (v_features(v_p(0), v_p(1), v_p(2), 0) > v_udf_threshold)
					continue;

				Eigen::Vector3d p = v_p.cast<double>().array() / (res - 1) * 2 - 1;
				Eigen::Vector3d n(
					v_features(v_p(0), v_p(1), v_p(2), 1),
					v_features(v_p(0), v_p(1), v_p(2), 2),
					v_features(v_p(0), v_p(1), v_p(2), 3)
				);
				n.normalize();

				std::vector<Eigen::Vector3d> plr; // Points along the ray
				plr.emplace_back(p);

				// Two requirements
				// 1. The change of udf value is less than 4 * ss
				// 2. The angle of gradient is either 0 or 180

				int summed_flags = v_flags(v_p[0], v_p[1], v_p[2]);
				int sign_flag = 1;
				for (int direction = -1; direction <= 1; direction += 2)
				{
					double last_udf = v_features(v_p(0), v_p(1), v_p(2), 0);
					int steps = 1;
					while (true)
					{
						Eigen::Vector3d np = p + ss * n * steps * direction; // New point
						// Voxelize
						Eigen::Vector3i coords = to_voxel_coor(np, res);

						if (coords.maxCoeff() >= res || coords.minCoeff() < 0)
							break;

						const double udf = v_features(coords(0), coords(1), coords(2), 0);
						Eigen::Vector3d gradient(
							v_features(coords(0), coords(1), coords(2), 1),
							v_features(coords(0), coords(1), coords(2), 2),
							v_features(coords(0), coords(1), coords(2), 3)
						);
						gradient.normalize();

						if (udf > v_udf_threshold)
							break;
						else if (std::abs(udf - last_udf) > 4 * ss)
							break;
						else if (n.dot(gradient) < -0.99 && udf < ss)
						{
							sign_flag = -1;
						}
						else if ((sign_flag * n).dot(gradient) < 0.9)
						{
							// LOG(INFO) << n.dot(-gradient);
							break;
						}

						last_udf = udf;
						plr.push_back(np);
						summed_flags += v_flags(coords(0), coords(1), coords(2));
						steps++;
					}
				}

				if (summed_flags > plr.size() * threshold)
				{
					for (const auto& p : plr)
					{
						Eigen::Vector3i coords = to_voxel_coor(p, res);
						new_flags(coords(0), coords(1), coords(2)) = true;
					}
				}

				// Debug
				if (false)
				{
					std::vector<Point_3> points;
					for (const auto& p : plr)
						points.emplace_back(eigen_2_cgal_point(p));
					export_points("temp/temp.ply", points);
				}
			}
		}
	);
	
	return new_flags;
}

Eigen::Tensor<bool, 4> build_edge_connectivity(
	const Eigen::Tensor<bool, 3>& v_consistent_flags,
	const int half_window_size
)
{
	const int resolution = static_cast<int>(v_consistent_flags.dimension(0));
	Eigen::Tensor<bool, 4> connectivity(resolution, resolution, resolution, 26);

	#pragma omp parallel for
	for (int i = 0; i < resolution * resolution * resolution; ++i)
	{
		const int x = i / resolution / resolution;
		const int y = i / resolution % resolution;
		const int z = i % resolution;

		int iter = 0;
		for (int dx = -1; dx <= 1; ++dx)
			for (int dy = -1; dy <= 1; ++dy)
				for (int dz = -1; dz <= 1; ++dz)
				{
					if (dx == 0 && dy == 0 && dz == 0)
						continue;
					if (!check_range(x + dx, y + dy, z + dz, resolution))
					{
						connectivity(x, y, z, iter) = false;
						iter++;
						continue;
					}
					// Detect the connectivity through a cube
					bool can_reach = true;
					for (int x_window = -half_window_size; x_window <= half_window_size; ++x_window)
						for (int y_window = -half_window_size; y_window <= half_window_size; ++y_window)
							for (int z_window = -half_window_size; z_window <= half_window_size; ++z_window)
							{
								if (!check_range(x + x_window + dx, y + y_window + dy, z + z_window + dz, resolution))
									continue;
								can_reach = can_reach && !v_consistent_flags(
									x + x_window + dx, y + y_window + dy, z + z_window + dz);
							}
					connectivity(x, y, z, iter) = can_reach;
					iter++;
				}
	}

	return connectivity;
}

// #pragma optimize ("", off)
// #pragma optimize ("", on)