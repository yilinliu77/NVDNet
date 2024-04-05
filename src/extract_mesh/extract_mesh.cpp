#include "common_util.h"

#include "boundary_growing.h"
#include "tools.h"
#include "assemble.h"
#include "filling_holes.h"
#include "fitting.h"
#include "merge_shape.h"
#include "shape2d.h"
#include "shape3d.h"

#include <numeric>
#include <sstream>
#include <istream>
#include <fstream>
#include <omp.h>

#include <argparse/argparse.hpp>

#include <CGAL/edge_aware_upsample_point_set.h>
#include <CGAL/pca_estimate_normals.h>

#include <gp_Pln.hxx>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include "npy.hpp"
#include "cgal_tools.h"

#include <tbb/tbb.h>

#include "assemble_loops.h"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Face.hxx>
#include <BRepLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepBuilderAPI_MakeShell.hxx>
#include <BRep_Builder.hxx>
#include <StlAPI_Writer.hxx>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>

// #pragma optimize("", off)

std::tuple<Eigen::Tensor<double, 4>, Eigen::Tensor<bool, 3>, Eigen::Tensor<double, 3>> load_data(
	const fs::path& v_root, const std::string& v_prefix)
{
	std::vector<unsigned long> feat_shape;
	std::vector<unsigned long> flag_shape;

	bool fortran_order;
	std::vector<unsigned short> raw_data_feat;
	std::vector<unsigned char> raw_data_flag;

	// npy::LoadArrayFromNumpy("F:/GSP/training/00002976_feat.npy",
	// npy::LoadArrayFromNumpy("C:/repo/python/outputs/test_data/00000003_feat.npy",
	// 	feat_shape, fortran_order, raw_data_feat);
	// npy::LoadArrayFromNumpy("C:/repo/python/outputs/test_data/00000003_gt.npy",
	// 	flag_shape, fortran_order, raw_data_flag);
	// HighFive::File file(v_root, HighFive::File::ReadOnly);

	const std::string feat_file = (v_root / (v_prefix + "_feat.npy")).string();
	const std::string flag_file = (v_root / (v_prefix + "_flag.npy")).string();
	if (!fs::exists(feat_file))
	{
		LOG(ERROR) << "Feature do not exist";
		exit(0);
	}
	if (!fs::exists(flag_file))
	{
		LOG(ERROR) << "Flag do not exist";
		exit(0);
	}

	// const auto& dataset_flag = file.getDataSet(v_prefix + "_flag");
	// raw_data_flag = dataset_flag.read<std::vector<unsigned char>>();
	// const auto& dataset_feat = file.getDataSet(v_prefix + "_feat");
	// raw_data_feat = dataset_feat.read<std::vector<unsigned short>>();


	npy::LoadArrayFromNumpy(feat_file,
	                        feat_shape, fortran_order, raw_data_feat);
	if (feat_shape[4] != 3)
		throw;
	Eigen::Tensor<double, 5, Eigen::RowMajor> feat_raw = Eigen::TensorMap<Eigen::Tensor<
		unsigned short, 5, Eigen::RowMajor>>(raw_data_feat.data(),
		                                     512, 32, 32, 32, 3).cast<double>();

	Eigen::Tensor<double, 7, Eigen::RowMajor> feat_reshaped_ = feat_raw.reshape(
		Eigen::array<int, 7>{8, 8, 8, 32, 32, 32, 3}).shuffle(
		Eigen::array<int, 7>{0, 3, 1, 4, 2, 5, 6});
	Eigen::Tensor<double, 4, Eigen::RowMajor> feat_reshaped = Eigen::TensorMap<Eigen::Tensor<
		double, 4, Eigen::RowMajor>>(
		feat_reshaped_.data(), 256, 256, 256, 3);

	npy::LoadArrayFromNumpy(flag_file,
	                        flag_shape, fortran_order, raw_data_flag);

	Eigen::Tensor<bool, 4, Eigen::RowMajor> flag_raw = Eigen::TensorMap<Eigen::Tensor<
		unsigned char, 4, Eigen::RowMajor>>(
		raw_data_flag.data(),
		512, 32, 32, 32).cast<int>().cast<bool>();

	Eigen::Tensor<bool, 6, Eigen::RowMajor> flag_reshaped_ = flag_raw.reshape(
		Eigen::array<int, 6>{8, 8, 8, 32, 32, 32}).shuffle(
		Eigen::array<int, 6>{0, 3, 1, 4, 2, 5});
	Eigen::Tensor<bool, 3, Eigen::RowMajor> flag_reshaped = Eigen::TensorMap<Eigen::Tensor<bool, 3, Eigen::RowMajor>>(
		flag_reshaped_.data(), 256, 256, 256);

	long long resolution, multiplier;
	if (feat_shape[1] != flag_shape[1])
	{
		throw;
		if (feat_shape[1] < flag_shape[1] || feat_shape[1] % flag_shape[1] != 0)
		{
			LOG(ERROR) << "Wrong resolution";
			throw;
		}
		resolution = feat_shape[1];
		multiplier = resolution / flag_shape[1];
	}
	else
	{
		resolution = 256;
		multiplier = 1;
	}

	long long num_feature = feat_shape[3];

	Eigen::Tensor<double, 4> gradients(3, resolution, resolution, resolution);
	Eigen::Tensor<bool, 3> consistent_flags(resolution, resolution, resolution);
	Eigen::Tensor<double, 3> udf(resolution, resolution, resolution);

	auto gradients_ptr = gradients.data();
	auto consistent_flags_ptr = consistent_flags.data();
	auto udf_flags_ptr = udf.data();

	auto timer = recordTime();
	const long long resolution_2 = resolution * resolution;
	// #pragma omp parallel
	const long long chunk = 32;
	const long long chunk_3 = chunk * chunk * chunk;
	const long long num_chunks_1d = resolution / chunk;
	const long long num_chunks_3d = num_chunks_1d * num_chunks_1d * num_chunks_1d;

#pragma omp parallel for
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
			{
				// Gradient direction
				double phi = feat_reshaped(x, y, z, 1);
				double theta = feat_reshaped(x, y, z, 2);
				phi = phi / 65535 * (M_PI * 2);
				theta = theta / 65535 * (M_PI * 2);

				Eigen::Vector3d gradient(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta),
				                         std::cos(theta));
				gradient = gradient.normalized();

				gradients_ptr[0 + z * 3 * resolution_2 + y * 3 * resolution + x * 3] = gradient[0];
				gradients_ptr[1 + z * 3 * resolution_2 + y * 3 * resolution + x * 3] = gradient[1];
				gradients_ptr[2 + z * 3 * resolution_2 + y * 3 * resolution + x * 3] = gradient[2];

				// Distance
				double dis = feat_reshaped(x, y, z, 0);
				dis = dis / 65535 * 2;
				udf_flags_ptr[z * resolution_2 + y * resolution + x] = dis;

				// Flags
				const bool consistent_flag = flag_reshaped(x, y, z);
				consistent_flags_ptr[z * resolution_2 + y * resolution + x] = consistent_flag;
			}

	profileTime(timer, "Extract: ", true);
	return {gradients, consistent_flags, udf};
}

std::tuple<Eigen::Tensor<double, 4>, Eigen::Tensor<bool, 3>> load_data_dir(
	const fs::path& v_root, const std::string& v_prefix, const int flag_mode)
{
	// Read data
	std::vector<unsigned long> feat_shape;
	std::vector<unsigned long> flag_shape;

	bool fortran_order;
	std::vector<float> raw_data_feat;
	std::vector<unsigned char> raw_data_flag;

	const std::string feat_file = (v_root / (v_prefix + "_feat.npy")).string();
	std::string flag_file;
	if (flag_mode == 0)
		flag_file = (v_root / (v_prefix + "_flag.npy")).string();
	else if (flag_mode == 1)
		flag_file = (v_root / (v_prefix + "_pred.npy")).string();
	else if (flag_mode == 2)
		flag_file = (v_root / (v_prefix + "_gt.npy")).string();
	else
	{
		LOG(ERROR) << "Incorrect flag mode";
		throw;
	}
	if (!fs::exists(feat_file))
	{
		LOG(ERROR) << "Feature do not exist";
		exit(0);
	}
	if (!fs::exists(flag_file))
	{
		LOG(ERROR) << "Flag do not exist";
		exit(0);
	}

	npy::LoadArrayFromNumpy(feat_file, feat_shape, fortran_order, raw_data_feat);
	npy::LoadArrayFromNumpy(flag_file, flag_shape, fortran_order, raw_data_flag);
	if (feat_shape[3] != 7)
	{
		LOG(ERROR) << "Incorrect shape";
		throw;
	}
	const int resolution = feat_shape[0];

	Eigen::Tensor<double, 4, Eigen::RowMajor> feat_reshaped = Eigen::TensorMap<Eigen::Tensor<
		float, 4, Eigen::RowMajor>>(
		raw_data_feat.data(), resolution, resolution, resolution, 7).cast<double>();
	Eigen::Tensor<bool, 3, Eigen::RowMajor> flag_reshaped = Eigen::TensorMap<Eigen::Tensor<
		unsigned char, 3, Eigen::RowMajor>>(
		raw_data_flag.data(), resolution, resolution, resolution).cast<int>().cast<bool>();

	if (resolution != flag_shape[1])
		throw;

	auto timer = recordTime();

	Eigen::array<int, 4> shuffle{3, 2, 1, 0};
	Eigen::Tensor<double, 4> features = feat_reshaped.swap_layout().shuffle(shuffle);

	Eigen::array<int, 3> shuffle3{2, 1, 0};
	Eigen::Tensor<bool, 3> consistent_flags = flag_reshaped.swap_layout().shuffle(shuffle3);

	profileTime(timer, "Extract: ", true);
	return {features, consistent_flags};
}

void debug_shapes(const std::vector<std::shared_ptr<Shape>>& v_shapes, const fs::path& v_path,
                  const std::string& v_type = "")
{
	checkFolder(v_path);
	Point_set inlier_points;
	Point_set resampled_points;
	auto index_map_inlier = inlier_points.add_property_map<int>("index").first;
	auto index_map_resampled = resampled_points.add_property_map<int>("index").first;
	const auto color_map = get_color_table_bgr2();
	for (int i = 0; i < v_shapes.size(); ++i)
	{
		if (!v_type.empty() && v_shapes[i]->type != v_type)
			continue;

		for (int j = 0; j < v_shapes[i]->inliers.size(); ++j)
			index_map_inlier[*inlier_points.insert(eigen_2_cgal_point(v_shapes[i]->inliers[j]))] = i;

		Point_set local_points = v_shapes[i]->sample_parametric(10000);
		const auto color = color_map[i % color_map.size()];
		local_points.add_property_map<uchar>("red", color[2]);
		local_points.add_property_map<uchar>("green", color[1]);
		local_points.add_property_map<uchar>("blue", color[0]);
		CGAL::IO::write_point_set((v_path / (std::to_string(i) + "_" + v_shapes[i]->detail_type + ".ply")).string(),
		                          local_points);
		for (int j = 0; j < local_points.size(); ++j)
			index_map_resampled[*resampled_points.insert(local_points.point(j))] = i;
	}
	colorize_point_set(inlier_points);
	colorize_point_set(resampled_points);

	CGAL::IO::write_point_set((v_path / "segmented_inliers.ply").string(), inlier_points);
	CGAL::IO::write_point_set((v_path / "segmented_resampled.ply").string(), resampled_points);
}

// #pragma optimize("", off)
// #pragma optimize("", on)
void extract_shapes(const std::vector<std::shared_ptr<Shape>>& v_shapes, const Eigen::MatrixXi& adj_matrix,
	const fs::path& v_path, const double epsilon = 0.005, const double alpha_value = 0.0004,
	const bool check_curve_before_adding=true)
{
	checkFolder(v_path);

	const auto color_map = get_color_table_bgr2();

	std::vector<Point_3> points;
	std::vector<std::vector<size_t>> indices;
	std::vector<CGAL::Color> colors;
	std::vector<int> index_map;

	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const tbb::blocked_range<int>& r0)
      {
          for (int i = r0.begin(); i < r0.end(); ++i)
          {
              if (v_shapes[i]->type != "curve")
                  continue;
              v_shapes[i]->get_inliers(v_shapes[i]->cluster.surface_points, epsilon);
              if (v_shapes[i]->inliers.empty())
                  continue;
              v_shapes[i]->find_boundary();
          }
      }
	);
	std::mutex mutex;
	tbb::parallel_for(tbb::blocked_range<int>(0, v_shapes.size()), [&](const tbb::blocked_range<int>& r0)
	{
		for (int i_surface = r0.begin(); i_surface < r0.end(); ++i_surface)
		{
			if (v_shapes[i_surface]->type != "surface")
				continue;

			const auto local_shape = dynamic_pointer_cast<Shape3D>(v_shapes[i_surface]);

			Surface_mesh local_mesh;
			Point_set boundaries;
			for (int i_curve = 0; i_curve < v_shapes.size(); ++i_curve)
			{
				if (v_shapes[i_curve]->type != "curve" || adj_matrix(i_surface, i_curve) == 0)
					continue;
				Point_set a = v_shapes[i_curve]->sample_parametric(10000);
				boundaries += a;
			}

			std::vector<Point_2> parametrics;
			for (int j = 0; j < local_shape->cluster.surface_points.size(); ++j)
				if (local_shape->distance(local_shape->cluster.surface_points[j]) < epsilon)
					parametrics.emplace_back(
						eigen_2_cgal_point(local_shape->get_parametric(local_shape->cluster.surface_points[j])));

			if (!check_curve_before_adding)
			{
				for (const Point_3& item : boundaries.points())
					parametrics.emplace_back(eigen_2_cgal_point(local_shape->get_parametric(cgal_2_eigen_point<double>(item))));
			}
			else
			{
				for (int i_curve = 0; i_curve < v_shapes.size(); ++i_curve)
				{
					if (v_shapes[i_curve]->type != "curve" || adj_matrix(i_surface, i_curve) == 0)
						continue;
					for (int i_point = 0; i_point < v_shapes[i_curve]->cluster.surface_points.size(); ++i_point)
						if (v_shapes[i_surface]->inlier_distance(v_shapes[i_curve]->cluster.surface_points[i_point]) < epsilon)
							parametrics.emplace_back(eigen_2_cgal_point(
								dynamic_pointer_cast<Shape3D>(v_shapes[i_surface])->get_parametric(
									v_shapes[i_curve]->cluster.surface_points[i_point])));
				}
			}
			

			if (local_shape->detail_type == "cylinder")
			{
				double min_theta=9999., max_theta=-9999.;
				double min_phi=9999., max_phi = -9999.;
				for(const auto& param: parametrics)
				{
					min_phi = std::min(min_phi, param.y());
					max_phi = std::max(max_phi, param.y());
					min_theta = std::min(min_theta, param.x());
					max_theta = std::max(max_theta, param.x());

				}
				const int num_samples = parametrics.size();
				if (min_theta < 1e-2 && max_theta > 1 - 1e-2)
					for (int i_sample=0; i_sample < num_samples;++i_sample)
					{
						if (parametrics[i_sample].x()<1e-2)
							parametrics.emplace_back(0, parametrics[i_sample].y());
						else if(parametrics[i_sample].x() > 1 - 1e-2)
							parametrics.emplace_back(1., parametrics[i_sample].y());
					}
			}

			Alpha_shape_2 as(
				parametrics.begin(), parametrics.end(), alpha_value, Alpha_shape_2::GENERAL);


			for (auto avit = as.finite_faces_begin(),
			          avit_end = as.finite_faces_end();
			     avit != avit_end; ++avit)
			{
				if (as.classify(avit) == Alpha_shape_2::EXTERIOR)
					continue;

				Point_2 p1 = as.triangle(avit).vertex(0);
				Point_2 p2 = as.triangle(avit).vertex(1);
				Point_2 p3 = as.triangle(avit).vertex(2);
				Point_3 p1_3d = eigen_2_cgal_point(local_shape->get_cartesian(Eigen::Vector2d(p1.x(), p1.y())));
				Point_3 p2_3d = eigen_2_cgal_point(local_shape->get_cartesian(Eigen::Vector2d(p2.x(), p2.y())));
				Point_3 p3_3d = eigen_2_cgal_point(local_shape->get_cartesian(Eigen::Vector2d(p3.x(), p3.y())));

				auto pp1 = local_mesh.add_vertex(p1_3d);
				auto pp2 = local_mesh.add_vertex(p2_3d);
				auto pp3 = local_mesh.add_vertex(p3_3d);

				auto face = local_mesh.add_face(
					pp1,
					pp2,
					pp3
				);
			}

			auto red_map = local_mesh.add_property_map<Surface_mesh::Face_index, uchar>(
				"red", color_map[i_surface % color_map.size()][2]).first;
			auto green_map = local_mesh.add_property_map<Surface_mesh::Face_index, uchar>(
				"green", color_map[i_surface % color_map.size()][1]).first;
			auto blue_map = local_mesh.add_property_map<Surface_mesh::Face_index, uchar>(
				"blue", color_map[i_surface % color_map.size()][0]).first;

			mutex.lock();
			const int cur_num_face = points.size();
			for (const auto& v : local_mesh.vertices())
				points.emplace_back(local_mesh.point(v));
			for (const auto& f : local_mesh.faces())
			{
				auto he = local_mesh.halfedge(f);
				auto v1 = local_mesh.source(he);
				auto v2 = local_mesh.target(he);
				auto v3 = local_mesh.target(local_mesh.next(he));
				indices.emplace_back(std::vector<size_t>{v1 + cur_num_face, v2 + cur_num_face, v3 + cur_num_face});
				colors.emplace_back(red_map[f],
				                    green_map[f],
				                    blue_map[f]
				);
				index_map.emplace_back(i_surface);
			}
			mutex.unlock();
			CGAL::IO::write_PLY((v_path / (std::to_string(i_surface) + "_mesh.ply")).string(), local_mesh);
		}
	});

	CGAL::Polygon_mesh_processing::merge_duplicate_points_in_polygon_soup(points,
	                                                                      indices);
	// Surface_mesh total_mesh;
	// auto red_map = total_mesh.add_property_map<Surface_mesh::Face_index, uchar>("red", '0').first;
	// auto green_map = total_mesh.add_property_map<Surface_mesh::Face_index, uchar>("green", '0').first;
	// auto blue_map = total_mesh.add_property_map<Surface_mesh::Face_index, uchar>("blue", '0').first;
	//
	// std::vector<Surface_mesh::Vertex_index> vd(points.size());
	// for(int i = 0; i < points.size(); ++i)
	// 	vd[i]=total_mesh.add_vertex(points[i]);
	// for (int i = 0; i < indices.size(); ++i)
	// {
	// 	auto face = total_mesh.add_face(
	// 		vd[indices[i][0]],
	// 		vd[indices[i][1]],
	// 		vd[indices[i][2]]
	// 	);
	// 	// red_map[face] = colors[i][0];
	// 	// green_map[face] = colors[i][1];
	// 	// blue_map[face] = colors[i][2];
	// }

	std::vector<CGAL::Color> dummy_colors(points.size());

	// Write ply mesh with face color in binary format
	std::ofstream out((v_path / "0total_mesh.ply").string(), std::ios::binary);
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "element vertex " << points.size() << "\n";
	out << "property double x\n";
	out << "property double y\n";
	out << "property double z\n";
	out << "element face " << indices.size() << "\n";
	out << "property list uchar int vertex_index\n";
	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "property int primitive_index\n";
	out << "end_header\n";
	for (int i = 0; i < points.size(); ++i)
	{
		double x = points[i].x();
		double y = points[i].y();
		double z = points[i].z();
		out.write((char*)&x, sizeof(double));
		out.write((char*)&y, sizeof(double));
		out.write((char*)&z, sizeof(double));
	}
	for (int i = 0; i < indices.size(); ++i)
	{
		unsigned char n = 3;
		out.write((char*)&n, sizeof(unsigned char));
		int v1 = indices[i][0];
		int v2 = indices[i][1];
		int v3 = indices[i][2];
		out.write((char*)&v1, sizeof(int));
		out.write((char*)&v2, sizeof(int));
		out.write((char*)&v3, sizeof(int));
		out.write((char*)&colors[i][0], sizeof(unsigned char));
		out.write((char*)&colors[i][1], sizeof(unsigned char));
		out.write((char*)&colors[i][2], sizeof(unsigned char));
		out.write((char*)&index_map[i], sizeof(int));
	}
	out.close();
}


void process_item(
	const std::string& v_prefix,
	const fs::path& data_root_dir,
	const argparse::ArgumentParser& v_args
)
{
	std::vector<double> time_statics(10, 0.);
	auto timer = recordTime();
	const bool is_profile_time = true;

	const fs::path& global_output_dir = v_args.get<std::string>("--output_dir");

	const bool read_cache = v_args.get<bool>("--read_cache");
	const bool only_evaluate = v_args.get<bool>("--only_evaluate");

	fs::path output_dir = global_output_dir / v_prefix;
	if (read_cache)
	{
		safeCheckFolder(output_dir);
		safeCheckFolder(output_dir / "summary");
		safeCheckFolder(output_dir / "eval");
	}
	else
	{
		checkFolder(output_dir);
		safeCheckFolder(output_dir / "summary");
		checkFolder(output_dir / "eval");
	}

	Eigen::Tensor<double, 4> features; // (256,256,256, 7)
	Eigen::Tensor<bool, 3> is_voronoi_boundary; // (256,256,256)

	const int read_flag = v_args.get<int>("--flag_mode");
	std::tie(features, is_voronoi_boundary) = load_data_dir(
		data_root_dir,
		v_prefix,
		read_flag

	);

	int resolution = features.dimension(1);
	const double udf_threshold = 0.07;

	Eigen::Tensor<double, 4> surface_points(resolution, resolution, resolution, 6);

	// Visualize raw data
	{
		// Build surface points
		tbb::parallel_for(tbb::blocked_range<int>(0, resolution), [&](const tbb::blocked_range<int>& r0)
		{
			for (int x = r0.begin(); x < r0.end(); ++x)
			{
				for (int y = 0; y < resolution; ++y)
					for (int z = 0; z < resolution; ++z)
					{
						Eigen::Vector3d position(
							x / 255. * 2 - 1,
							y / 255. * 2 - 1,
							z / 255. * 2 - 1
						);
						Eigen::Vector3d dir(
							features(x, y, z, 1),
							features(x, y, z, 2),
							features(x, y, z, 3)
						);
						Eigen::Vector3d normal(
							features(x, y, z, 4),
							features(x, y, z, 5),
							features(x, y, z, 6)
						);
						dir.normalize();
						normal.normalize();
						position = position + features(x, y, z, 0) * dir;
						surface_points(x, y, z, 0) = position[0];
						surface_points(x, y, z, 1) = position[1];
						surface_points(x, y, z, 2) = position[2];
						surface_points(x, y, z, 3) = normal[0];
						surface_points(x, y, z, 4) = normal[1];
						surface_points(x, y, z, 5) = normal[2];
					}
			}
		});

		if (!only_evaluate)
		{
			Point_set total_unsegmented;
			total_unsegmented.resize(resolution * resolution * resolution);
			for (int x = 0; x < resolution; ++x)
				for (int y = 0; y < resolution; ++y)
					for (int z = 0; z < resolution; ++z)
					{
						Point_3 original_point(
							surface_points(x, y, z, 0),
							surface_points(x, y, z, 1),
							surface_points(x, y, z, 2)
						);
						total_unsegmented.point(
							x * resolution * resolution + y * resolution + z) = original_point;
					}
			CGAL::IO::write_point_set((output_dir / "summary/1_total_unsegmented.ply").string(), total_unsegmented);
		}
	}
	time_statics[0] += profileTime(timer, "12 IO", is_profile_time);

	// 3. Some preprocessing
	if (!only_evaluate)
		export_points((output_dir / "summary/2_init_boundary.ply").string(), is_voronoi_boundary, resolution);
	Eigen::VectorXi bounds;
	std::tie(is_voronoi_boundary, bounds) = mark_boundary(is_voronoi_boundary, features, udf_threshold);
	if (!only_evaluate)
		export_points((output_dir / "summary/5_marked_boundary.ply").string(), is_voronoi_boundary, resolution,
		              features, udf_threshold);

	Eigen::Tensor<bool, 3> is_valid_global(resolution, resolution, resolution);
	is_valid_global.setConstant(true);
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
				if (features(x, y, z, 0) > udf_threshold)
					is_valid_global(x, y, z) = false;

	time_statics[1] += profileTime(timer, "1 preprocessing", is_profile_time);

	// Start to recover
	std::string shape_cache = (output_dir / "summary/shape_cache").string();
	std::vector<std::shared_ptr<Shape>> shapes;
	Point_set surface_boundary;
	if (fs::exists(shape_cache) && read_cache)
	{
		LOG(INFO) << "Found existing shapes cache " << shape_cache;
		shapes.clear();
		std::ifstream ifs(shape_cache, std::ios::binary | std::ios::in);
		boost::archive::binary_iarchive oa(ifs);
		oa >> shapes;
		ifs.close();
		// colorize_output_points(shapes);
		surface_boundary = get_boundaries(shapes);
	}
	else
	{
		const int dilate_radius = v_args.get<int>("--dilate_radius");
		const double alpha_value = v_args.get<float>("--alpha_value");
		// 4. Filling holes
		if (false)
		{
			std::string points_cache = "temp/summary/6_alpha_shape_boundary.ply";

			std::vector<Triangle_3> triangles;
			if (fs::exists(points_cache) && read_cache)
			{
				Point_set dilated_boundary;
				CGAL::IO::read_point_set(points_cache, dilated_boundary);
#pragma omp parallel for
				for (int i = 0; i < dilated_boundary.size(); ++i)
				{
					Eigen::Vector3d sp = cgal_2_eigen_point<double>(dilated_boundary.point(i));
					sp = (sp.array() + 1) / 2 * (resolution - 1);
					is_voronoi_boundary(
						(int)std::round(sp[0]),
						(int)std::round(sp[1]),
						(int)std::round(sp[2])
					) = true;
				}
			}
			else
			{
				Point_set boundary_points;
				for (int x = 0; x < resolution; ++x)
					for (int y = 0; y < resolution; ++y)
						for (int z = 0; z < resolution; ++z)
						{
							if (!is_voronoi_boundary(x, y, z))
								continue;

							if (!within_bounds(x, y, z, bounds))
								continue;

							Eigen::Vector3d cur_pos(x, y, z);
							cur_pos = cur_pos / (resolution - 1) * 2 - Eigen::Vector3d::Ones();

							boundary_points.insert(eigen_2_cgal_point(cur_pos));
						}
				CGAL::IO::write_point_set("temp/points_before_filling_holes.ply", boundary_points);
				triangles = filling_holes(boundary_points,
				                          alpha_value
				);
			}

			is_voronoi_boundary = rebuild_flags(triangles, is_voronoi_boundary, resolution * resolution * 2);
			export_points(points_cache, is_voronoi_boundary, resolution, features, udf_threshold);
			is_voronoi_boundary = dilate(is_voronoi_boundary, resolution, dilate_radius);
			export_points("temp/summary/7_dilated_boundary.ply", is_voronoi_boundary, resolution, features,
			              udf_threshold);
		}
		if (true)
		{
			std::string points_cache = (output_dir / "summary/6_alpha_shape_boundary.ply").string();

			std::vector<Triangle_3> triangles;
			if (fs::exists(points_cache) && read_cache)
			{
				Point_set dilated_boundary;
				CGAL::IO::read_point_set(points_cache, dilated_boundary);
				// #pragma omp parallel for
				for (int i = 0; i < dilated_boundary.size(); ++i)
				{
					Eigen::Vector3d sp = cgal_2_eigen_point<double>(dilated_boundary.point(i));
					sp = (sp.array() + 1) / 2 * (resolution - 1);
					is_voronoi_boundary(
						(int)std::round(sp[0]),
						(int)std::round(sp[1]),
						(int)std::round(sp[2])
					) = true;
				}
			}
			else
			{
				is_voronoi_boundary = dilate(is_voronoi_boundary, resolution, dilate_radius);

				for (int i = 0; i < 1; ++i)
				{
					is_voronoi_boundary = dilate_along_the_ray(
						features,
						udf_threshold,
						is_voronoi_boundary,
						0.5
					);
				}
				// edge_flag = dilate_according_to_gradients(edge_flag, features, 3, udf_threshold);
			}
			if (!only_evaluate)
				export_points(points_cache, is_voronoi_boundary, resolution, features, udf_threshold);
		}
		time_statics[2] += profileTime(timer, "2 Filling holes", is_profile_time);

		auto connectivity = build_edge_connectivity(is_voronoi_boundary, 0);

		const int num_cpus = 8;
		// region growing
		std::string region_cache = (output_dir / "summary/region_cache").string();
		std::vector<Cluster> clusters;
		{
			if (fs::exists(region_cache) && read_cache)
			{
				LOG(INFO) << "Found existing region growing cache " << region_cache;
				std::stringstream archive_stream;
				std::ifstream ifs(region_cache, std::ios::binary | std::ios::in);
				ifs >> archive_stream.rdbuf();
				ifs.close();
				boost::archive::binary_iarchive oa(archive_stream);
				oa >> clusters;
			}
			else
			{
				LOG(INFO) << ffmt("Start to do region growing");
				clusters = classify_points_region_growing(
					is_voronoi_boundary, connectivity, resolution, num_cpus, features, output_dir, only_evaluate);
				LOG(INFO) << ffmt("%d clusters") % clusters.size();

				if (!only_evaluate)
				{
					std::ostringstream archive_stream;
					boost::archive::binary_oarchive oa(archive_stream);
					oa << clusters;
					std::ofstream ofs(region_cache, std::ios::binary | std::ios::out);
					ofs << archive_stream.str();
					ofs.close();
				}

				LOG(INFO) << ffmt("Done saving");
			}
		}
		time_statics[3] += profileTime(timer, "3 Growing", is_profile_time);

		// (Optional) Segmentation
		if (false)
		{
			std::string points_file = "G:/Dataset/GSP/Results/BSpline/poisson1k.ply";
			Point_set points;
			CGAL::IO::read_point_set(points_file, points);
			const int num_points = points.size();
			std::vector<int> point_coords(num_points, 0);

			std::vector<int> cluster_ids(num_points, -1);
			tbb::parallel_for(tbb::blocked_range<int>(0, num_points), 
				[&](const auto& r0)
				{
					for (int i = r0.begin(); i < r0.end(); ++i)
					{
						Eigen::Vector3d pos = cgal_2_eigen_point<double>(points.point(i));
						Eigen::Vector3i coord = to_voxel_coor(pos, resolution);
						for(int i_cluster=0;i_cluster<clusters.size();++i_cluster)
						{
							const auto position = std::find(clusters[i_cluster].coords.begin(), clusters[i_cluster].coords.end(), coord);
							if (position != clusters[i_cluster].coords.end())
								cluster_ids[i] = i_cluster;
						}
					}
				});

			KdTreeHelper kdtree(points);

			// Give id to unsegmented points
			{
				while(true)
				{
					bool is_finished = true;
					for(int i=0;i<num_points;++i)
					{
						if (cluster_ids[i] != -1)
							continue;

						is_finished = false;

						const auto results = kdtree.search_k_neighbour(
							cgal_2_eigen_point<float>(points.point(i)), 100);
						for(int j=1;j<results.first.size();++j)
						{
							if (results.second[j] > 0.02)
								break;

							if (cluster_ids[results.first[j]] != -1)
							{
								cluster_ids[i] = cluster_ids[results.first[j]];
								break;
							}
						}
					}
					if (is_finished)
						break;
				}
			}

			auto color_map = points.add_property_map("primitive_index", 0).first;
			for (int i = 0; i < cluster_ids.size(); ++i)
			{
				color_map[i] = cluster_ids[i];
				std::cout << cluster_ids[i] << " ";
			}

			colorize_point_set(points, "primitive_index");
			CGAL::IO::write_point_set("G:/Dataset/GSP/Results/BSpline/clustered_points.ply", points);
		}

		// 5. Fitting
		const double fitting_epsilon = v_args.get<float>("--fitting_epsilon");
		const bool fallback_ransac = v_args.get<bool>("--fallback_ransac");
		const int num_fitting_points = v_args.get<int>("--num_fitting_points");
		{
			std::string fitting_cache = (output_dir / "summary/fitting_cache").string();
			if (fs::exists(fitting_cache) && read_cache)
			{
				LOG(INFO) << "Found existing fitting cache " << fitting_cache;
				std::stringstream archive_stream;
				std::ifstream ifs(fitting_cache, std::ios::binary | std::ios::in);
				ifs >> archive_stream.rdbuf();
				ifs.close();
				boost::archive::binary_iarchive oa(archive_stream);
				oa >> shapes;
				// colorize_output_points(shapes);
			}
			else
			{
				LOG(INFO) << ffmt("Start to do fitting");
				shapes = fitting(clusters, fitting_epsilon, num_fitting_points);
				if (!only_evaluate)
				{
					checkFolder(output_dir / "fitting");
					debug_shapes(shapes, output_dir / "fitting");
				}
				LOG(INFO) << ffmt("%d shapes") % shapes.size();

				// Output
				if (!only_evaluate)
				{
					std::ostringstream archive_stream;
					boost::archive::binary_oarchive oa(archive_stream);
					oa << shapes;
					std::ofstream ofs(fitting_cache, std::ios::binary | std::ios::out);
					ofs << archive_stream.str();
					ofs.close();
					// colorize_output_points(shapes);
				}
				LOG(INFO) << ffmt("Done saving");
			}
		}
		time_statics[4] += profileTime(timer, "4 fitting", is_profile_time);

		// 6. Expanding
		const bool is_restricted = v_args.get<bool>("--restricted");
		const int max_num_points_refit = v_args.get<int>("--max_num_points_refit");
		{
			LOG(INFO) << ffmt("Start to merge");
			shapes = merge_shape(shapes, fitting_epsilon, resolution);
			LOG(INFO) << ffmt("%d shapes after merging") % shapes.size();

			if (!only_evaluate)
				debug_shapes(shapes, output_dir / "surface_after_merging");

			const int num_expanding = 3;
			for (int i = 0; i < num_expanding; ++i)
			{
				LOG(INFO) << ffmt("Start the %d expand") % i;
				boundary_grow_surface(surface_points, shapes, fitting_epsilon * 5, is_valid_global,
					// boundary_grow_surface(surface_points, shapes, 0.01, is_valid_global,
				                      is_voronoi_boundary, is_restricted, max_num_points_refit);
				LOG(INFO) << ffmt("Merge");
				shapes = merge_shape(shapes, fitting_epsilon, resolution);
				LOG(INFO) << ffmt("%d shapes after merging") % shapes.size();
			}
			if (!only_evaluate)
				debug_shapes(shapes, output_dir / "surface_after_expanding");

			// Post processing
			if (fallback_ransac)
			{
				LOG(INFO) << "Record the unused points";
				Eigen::Tensor<bool, 3> used_flags(resolution, resolution, resolution);
				used_flags.setConstant(false);
				const int original_shape_size = shapes.size();
				for (int i_shape = shapes.size() - 1; i_shape >= 0; --i_shape)
				{
					if (shapes[i_shape]->type != "surface")
						continue;

					auto shape = dynamic_pointer_cast<Shape3D>(shapes[i_shape]);
					if (shape->area_sum < 1e-5)
						shapes.erase(shapes.begin() + i_shape);

					for (const auto& item : shape->cluster.coords)
						used_flags(item[0], item[1], item[2]) = true;
				}
				LOG(INFO) << "Filter out " << original_shape_size - shapes.size() << " shapes with zero area";

				for (int x = 0; x < resolution; ++x)
					for (int y = 0; y < resolution; ++y)
						for (int z = 0; z < resolution; ++z)
						{
							if (features(x, y, z, 0) > udf_threshold)
								used_flags(x, y, z) = true;
						}

				Cluster cluster;
				for (int x = 0; x < resolution; ++x)
					for (int y = 0; y < resolution; ++y)
						for (int z = 0; z < resolution; ++z)
							if (used_flags(x, y, z) == false)
							{
								cluster.coords.emplace_back(x, y, z);
								cluster.surface_points.emplace_back(
									surface_points(x, y, z, 0),
									surface_points(x, y, z, 1),
									surface_points(x, y, z, 2)
								);
								cluster.query_points.emplace_back(
									surface_points(x, y, z, 0),
									surface_points(x, y, z, 1),
									surface_points(x, y, z, 2)
								);
							}
				export_points((output_dir / "fitting/0remaining_points.ply").string(), cluster.surface_points);

				LOG(INFO) << "Final RANSAC on remaining points";
				{
					while (cluster.surface_points.size() > 20)
					{
						std::shared_ptr<Shape> shape = fall_back_ransac(cluster, fitting_epsilon);
						if (shape == nullptr)
							break;
						else
						{
							Cluster remain;
							shape->cluster.surface_points.clear();
							for (int i = 0; i < cluster.surface_points.size(); ++i)
							{
								if (shape->distance(cluster.surface_points[i]) < fitting_epsilon)
								{
									shape->cluster.query_points.push_back(cluster.query_points[i]);
									shape->cluster.coords.push_back(cluster.coords[i]);
									shape->cluster.surface_points.push_back(cluster.surface_points[i]);
								}
								else
								{
									remain.query_points.push_back(cluster.query_points[i]);
									remain.coords.push_back(cluster.coords[i]);
									remain.surface_points.push_back(cluster.surface_points[i]);
								}
							}
							if (shape->cluster.surface_points.size() < 10)
								break;
							shape->get_inliers(shape->cluster.surface_points, fitting_epsilon);
							shape->find_boundary();

							cluster = remain;

							LOG(INFO) << ffmt("Fallback ransac found %s of %d points in cluster") % shape->detail_type %
								shape->cluster.surface_points.size();
							shapes.push_back(shape);
						}
					}
				}

				for (int i = 0; i < num_expanding; ++i)
				{
					LOG(INFO) << ffmt("Start the %d expand") % i;
					boundary_grow_surface(surface_points, shapes, fitting_epsilon * 2, is_valid_global,
					                      is_voronoi_boundary, is_restricted);
					LOG(INFO) << ffmt("Merge");
					shapes = merge_shape(shapes, fitting_epsilon, resolution);
					LOG(INFO) << ffmt("%d shapes after merging") % shapes.size();
				}
				if (!only_evaluate)
					debug_shapes(shapes, output_dir / "surface_after_expanding");
			}

			// Get boundary
			surface_boundary = get_boundaries(shapes);
			if (!only_evaluate)
				CGAL::IO::write_point_set((output_dir / "shape_boundary.ply").string(), surface_boundary);

			// Curve expanding
			if (!only_evaluate)
				debug_shapes(shapes, output_dir / "curve_before_expanding", "curve");

			for (int i = 0; i < num_expanding; ++i)
			{
				LOG(INFO) << ffmt("Start the %d curve expand") % i;
				// boundary_grow_curve(surface_points, shapes, fitting_epsilon * 5, is_valid_global, is_voronoi_boundary);
				boundary_grow_curve(surface_points, shapes, fitting_epsilon * 2, is_valid_global, surface_boundary);
				LOG(INFO) << ffmt("Merge");
				shapes = merge_shape(shapes, fitting_epsilon * 2, resolution);
				LOG(INFO) << ffmt("%d shapes after merging") % shapes.size();
			}
			if (!only_evaluate)
				debug_shapes(shapes, output_dir / "curve_after_expanding", "curve");

			// Vertex expanding
			for (int i = 0; i < num_expanding; ++i)
			{
				LOG(INFO) << ffmt("Start the %d vertex expand") % i;
				boundary_grow_vertex(surface_points, shapes, fitting_epsilon * 5, is_valid_global);
				LOG(INFO) << ffmt("Merge");
				shapes = merge_shape(shapes, fitting_epsilon * 5, resolution);
				LOG(INFO) << ffmt("%d shapes after merging") % shapes.size();
			}
			if (!only_evaluate)
				debug_shapes(shapes, output_dir / "vertex_after_expanding", "vertex");
			LOG(INFO) << ffmt("Done; We have %d shapes now") % shapes.size();

			// Output
			if (!only_evaluate)
			{
				std::ofstream ofs(shape_cache, std::ios::binary | std::ios::out);
				boost::archive::binary_oarchive oa(ofs);
				oa << shapes;
				ofs.close();
				// colorize_output_points(shapes);
			}
			LOG(INFO) << ffmt("Done saving");
		}
		Eigen::MatrixXi adj_matrix(shapes.size(), shapes.size());
		adj_matrix.setZero();
		// extract_shapes(shapes, adj_matrix, output_dir / "mesh", fitting_epsilon * 5, 0.0001);
		time_statics[5] += profileTime(timer, "5 expanding", is_profile_time);
	}

	// Build kdtree for inliers to accelerate the distance computation
	tbb::parallel_for(tbb::blocked_range<int>(0, shapes.size()), [&](const tbb::blocked_range<int>& r0)
		{
			for (int i = r0.begin(); i < r0.end(); ++i)
			{
				if (shapes[i]->inliers.empty() || shapes[i]->type != "surface")
					continue;
				auto shape = dynamic_pointer_cast<Shape3D>(shapes[i]);
				shape->m_kdtree_data = initialize_kd_data(shape->inliers);
				shape->m_kd_tree = initialize_kd_tree(shape->m_kdtree_data);
			}
		}
	);

	if (v_args.get<bool>("--remove_voronoi_curves"))
	{
		std::vector<std::shared_ptr<Shape>> new_shapes;
		for(const auto& item: shapes)
		{
			if (item->type == "curve" || item->type == "vertex")
				continue;
			new_shapes.push_back(item);
		}
		shapes = new_shapes;
	}

	const double common_points_threshold = v_args.get<float>("--common_points_threshold");
	const double shape_epsilon = v_args.get<float>("--shape_epsilon");
	Eigen::MatrixXi adj_matrix = assemble(
		shapes, resolution, surface_points, surface_boundary,
		common_points_threshold, shape_epsilon,
		output_dir,!only_evaluate);
	time_statics[6] += profileTime(timer, "6 Compute curves", is_profile_time);

	// 8. Handle vertex
	const double vertex_threshold = v_args.get<float>("--vertex_threshold");
	solve_vertex(shapes, vertex_threshold, resolution, adj_matrix, output_dir, !only_evaluate);
	// Brute force add vertex
	/*
	for (int i_shape = shapes.size() - 1; i_shape >= 0; --i_shape)
	{
		if (shapes[i_shape]->type != "curve")
			continue;

		const auto& shape = dynamic_pointer_cast<Shape2D>(shapes[i_shape]);
		const Eigen::Vector3d v0 = shape->get_cartesian(shape->min_t);
		const Eigen::Vector3d v1 = shape->get_cartesian(shape->max_t);

		std::shared_ptr<Shape1D> vertex0 = std::make_shared<Shape1D>();
		vertex0->type = "vertex";
		vertex0->vertex = v0;
		vertex0->inliers.push_back(v0);
		vertex0->cluster.surface_points.push_back(v0);

		std::shared_ptr<Shape1D> vertex1 = std::make_shared<Shape1D>();
		vertex1->type = "vertex";
		vertex1->vertex = v1;
		vertex1->inliers.push_back(v1);
		vertex1->cluster.surface_points.push_back(v1);

		shapes.push_back(vertex0);
		shapes.push_back(vertex1);
		adj_matrix.conservativeResize(adj_matrix.rows() + 2, adj_matrix.cols() + 2);
		adj_matrix.row(adj_matrix.rows() - 2).setZero();
		adj_matrix.row(adj_matrix.rows() - 1).setZero();
		adj_matrix.col(adj_matrix.cols() - 2).setZero();
		adj_matrix.col(adj_matrix.cols() - 1).setZero();
		adj_matrix(i_shape, adj_matrix.cols() - 1) = 1;
		adj_matrix(i_shape, adj_matrix.cols() - 2) = 1;
		adj_matrix(i_shape, adj_matrix.rows() - 1) = 1;
		adj_matrix(i_shape, adj_matrix.rows() - 2) = 1;
		adj_matrix( adj_matrix.cols() - 1, i_shape) = 1;
		adj_matrix( adj_matrix.cols() - 2, i_shape) = 1;
		adj_matrix( adj_matrix.rows() - 1, i_shape) = 1;
		adj_matrix( adj_matrix.rows() - 2, i_shape) = 1;
	}
	shapes = merge_shape(shapes, 0.02, resolution, adj_matrix,"vertex");
	*/
	time_statics[7] += profileTime(timer, "7 Compute vertices", is_profile_time);

	// 9. Handle loops
	assemble_loops(shapes, surface_points, surface_boundary);
	time_statics[8] += profileTime(timer, "8 Assemble", is_profile_time);

	// 10. Extract
	const bool check_curve_before_adding = v_args.get<bool>("--check_curve_before_adding");
	const double output_inlier_epsilon = v_args.get<float>("--output_inlier_epsilon");
	const double output_alpha_value = v_args.get<float>("--output_alpha_value");
	{
		LOG(INFO) << "Start to save results";

		Point_set vertices, curves, surfaces;
		auto vertices_index_map = vertices.add_property_map("primitive_index", 0).first;
		auto curves_index_map = curves.add_property_map("primitive_index", 0).first;
		auto surfaces_index_map = surfaces.add_property_map("primitive_index", 0).first;

		int num_vertices = 0, num_curves = 0, num_surfaces = 0;
		std::vector<int> id_vertex(shapes.size(), -1), id_curve(shapes.size(), -1), id_surface(shapes.size(), -1);
		for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		{
			// Write vertex
			if (shapes[i_shape]->type == "vertex")
			{
				vertices_index_map[*vertices.insert(
					eigen_2_cgal_point(dynamic_pointer_cast<Shape1D>(shapes[i_shape])->vertex))] = num_vertices;
				id_vertex[i_shape] = num_vertices;
				num_vertices += 1;
			}
			else if (shapes[i_shape]->type == "curve")
			{
				const auto shape = dynamic_pointer_cast<Shape2D>(shapes[i_shape]);
				// Compute length
				{
					int num_sampled_points;
					const double step = (shape->max_t - shape->min_t) / 10;
					Point_set point_set;
					std::vector<double> params(10);
					point_set.resize(10);
					for (int i = 0; i < point_set.size(); ++i)
					{
						const double t = shape->min_t + step * i;
						point_set.point(i) = eigen_2_cgal_point(shape->get_cartesian(t));
					}
					double length = 0.;
					for (int i = 0; i < point_set.size() - 1; ++i)
					{
						length += std::sqrt(CGAL::squared_distance(point_set.point(i), point_set.point(i + 1)));
					}

					num_sampled_points = std::ceil(length * 1000);
					num_sampled_points = std::max(1, num_sampled_points);

					Point_set p = shape->sample_parametric(num_sampled_points);
					for (const auto& item : p.points())
						curves_index_map[*curves.insert(item)] = num_curves;
					if (p.empty())
						throw;
				}
				
				id_curve[i_shape] = num_curves;
				num_curves += 1;
				// vertices_index_map[*vertices.insert(eigen_2_cgal_point(shape->get_cartesian(shape->min_t)))] = num_vertices++;
				// vertices_index_map[*vertices.insert(eigen_2_cgal_point(shape->get_cartesian(shape->max_t)))] = num_vertices++;
			}
			else
			{
				const auto shape = dynamic_pointer_cast<Shape3D>(shapes[i_shape]);
				Surface_mesh mesh;
				for (const auto& item : shape->m_boundary)
				{
					mesh.add_face(
						mesh.add_vertex(
							eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(0))))),
						mesh.add_vertex(
							eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(1))))),
						mesh.add_vertex(
							eigen_2_cgal_point(shape->get_cartesian(cgal_2_eigen_point<double>(item.vertex(2)))))
					);
				}
				double area = CGAL::Polygon_mesh_processing::area(mesh);
				const int num_points = std::ceil(area * 10000);
				Point_set p = shape->sample_parametric(num_points);
				for (const auto& item : p.points())
					surfaces_index_map[*surfaces.insert(item)] = num_surfaces;
				id_surface[i_shape] = num_surfaces;
				num_surfaces += 1;
			}
		}

		colorize_point_set(vertices, "primitive_index");
		colorize_point_set(curves, "primitive_index");
		colorize_point_set(surfaces, "primitive_index");

		CGAL::IO::write_point_set((output_dir / "eval/vertices.ply").string(), vertices);
		CGAL::IO::write_point_set((output_dir / "eval/curves.ply").string(), curves);
		CGAL::IO::write_point_set((output_dir / "eval/surfaces.ply").string(), surfaces);

		const int num_primitives = adj_matrix.rows();
		std::ofstream ofs((output_dir / "eval/adj_matrix.txt").string());
		ofs << "FE\n";
		for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		{
			if (shapes[i_shape]->type != "surface")
				continue;
			ofs << id_surface[i_shape];
			for (int ishape2 = 0; ishape2 < shapes.size(); ++ishape2)
			{
				if (shapes[ishape2]->type != "curve" || adj_matrix(i_shape, ishape2) == 0)
					continue;
				ofs << " " << id_curve[ishape2];
			}
			ofs << std::endl;
		}
		ofs << "EV\n";
		for (int i_shape = 0; i_shape < shapes.size(); ++i_shape)
		{
			if (shapes[i_shape]->type != "curve")
				continue;
			ofs << id_curve[i_shape];
			for (int ishape2 = 0; ishape2 < shapes.size(); ++ishape2)
			{
				if (shapes[ishape2]->type != "vertex" || adj_matrix(i_shape, ishape2) == 0)
					continue;
				ofs << " " << id_vertex[ishape2];
			}
			ofs << std::endl;
		}
		ofs.close();

		// Save curve and vertex point for visualization
		{
			checkFolder(output_dir / "viz_curve_and_vertex");
			for(int i_shape=0;i_shape<shapes.size();++i_shape)
			{
				std::vector<Eigen::Vector3d> points;
				if (shapes[i_shape]->detail_type == "line")
				{
					const auto shape = dynamic_pointer_cast<Shape2D>(shapes[i_shape]);
					double min_t = shape->min_t, max_t = shape->max_t;
					points.emplace_back(shape->get_cartesian(min_t));
					points.emplace_back(shape->get_cartesian(max_t));
					export_points(output_dir / "viz_curve_and_vertex" / (std::to_string(i_shape) + "_curve.ply"), points);
				}
				else if (shapes[i_shape]->detail_type == "circle")
				{
					const auto circle = dynamic_pointer_cast<MyCircle>(shapes[i_shape]);
					for (int i_bin = 0; i_bin < circle->bins.size(); ++i_bin)
					{
						if (circle->bins[i_bin].empty())
							continue;
						const int num_samples_local = 100;
						for (int i_sample = 0; i_sample < num_samples_local; ++i_sample)
						{
							const double t = ((double)i_sample / num_samples_local + i_bin) * circle->bin_range;
							points.emplace_back(circle->get_cartesian(t));
						}
					}
					export_points(output_dir / "viz_curve_and_vertex" / (std::to_string(i_shape) + "_curve.ply"), points);
				}
				else if (shapes[i_shape]->detail_type == "ellipse")
				{
					const auto ellipse = dynamic_pointer_cast<MyEllipse>(shapes[i_shape]);
					for (int i_bin = 0; i_bin < ellipse->bins.size(); ++i_bin)
					{
						if (ellipse->bins[i_bin].empty())
							continue;
						const int num_samples_local = 100;

						for (int i_sample = 0; i_sample < num_samples_local; ++i_sample)
						{
							const double t = ((double)i_sample / num_samples_local + i_bin) * ellipse->bin_range;
							points.emplace_back(ellipse->get_cartesian(t));
						}
					}
					export_points(output_dir / "viz_curve_and_vertex" / (std::to_string(i_shape) + "_curve.ply"), points);
				}
				else if (shapes[i_shape]->type == "vertex")
				{
					const auto shape = dynamic_pointer_cast<Shape1D>(shapes[i_shape]);
					points.emplace_back(shape->vertex);
					export_points(output_dir / "viz_curve_and_vertex" / (std::to_string(i_shape) + "_vertex.ply"), points);
				}
			}
		}

		extract_shapes(shapes, adj_matrix, output_dir / "mesh", 
			output_inlier_epsilon, output_alpha_value, check_curve_before_adding);

		// Save all
		{
			std::ofstream ofs1((output_dir / "shape_cache").string(), std::ios::binary | std::ios::out);
			boost::archive::binary_oarchive oa1(ofs1);
			oa1 << shapes;
			ofs1.close();

			std::ofstream ofs2((output_dir / "adj_cache").string(), std::ios::binary | std::ios::out);
			boost::archive::binary_oarchive oa2(ofs2);
			oa2 << adj_matrix;
			ofs2.close();
		}
		LOG(INFO) << "Done";
	}
	time_statics[9] += profileTime(timer, "9 IO", is_profile_time);
	return;
}

int main(int argc, char* argv[])
{
	if (false)
	{
		BRepLib::Precision(1e-2);
		// Define two circle with opposite orientation
		gp_Circ circle1(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1);
		gp_Circ circle2(gp_Ax2(gp_Pnt(0, 0, 1), gp_Dir(0, 0, -1)), 1);

		// Define a cylinder
		gp_Cylinder cy(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1);

		// Build edge
		BRepBuilderAPI_MakeEdge edge1(circle1);
		BRepBuilderAPI_MakeEdge edge2(gp_Pnt(1, 0, 0), gp_Pnt(1, 0, 1));
		BRepBuilderAPI_MakeEdge edge3(circle2, gp_Pnt(1, 0, 1), gp_Pnt(1, 0, 1));
		BRepBuilderAPI_MakeEdge edge4(gp_Pnt(1, 0, 1), gp_Pnt(1, 0, 0));

		BRepBuilderAPI_MakeWire makeWire1;
		makeWire1.Add(edge1);
		makeWire1.Add(edge2);
		makeWire1.Add(edge3);
		makeWire1.Add(edge4);
		TopoDS_Wire wire = makeWire1.Wire();

		// Limit cylinder by wired edges
		BRepBuilderAPI_MakeFace cylface(cy, wire, true);

		std::cout << cylface.Error() << std::endl;
		std::cout << cylface.IsDone() << std::endl;
		TopoDS_Face output = cylface.Face();

		BRepMesh_IncrementalMesh mesh(output, 5e-2, false, 0.5);
		StlAPI_Writer objWriter;
		bool write_flag = objWriter.Write(output, "1.stl");
		std::cout << write_flag << std::endl;
	}
	// main2();
	// omp_set_nested(1);
	// tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 16);
	tbb::global_control limit_thread(tbb::global_control::thread_stack_size, 32 * 1024 * 1024);
	LOG(INFO) << "enter the arguments: data_root prefix";
	argparse::ArgumentParser program("prepare_data_3d");
	{
		program.add_description("data_root output_root resolution id_start id_end is_log");

		// Path related
		program.add_argument("data_root").required();
		program.add_argument("--prefix").default_value(std::string(""));
		program.add_argument("--output_dir").default_value(std::string("temp"));
		program.add_argument("--flag_mode").default_value(0).scan<'i', int>();
		program.add_argument("--read_cache").implicit_value(true).default_value(false);
		program.add_argument("--only_evaluate").implicit_value(true).default_value(false);

		// Pre-processing related
		program.add_argument("--dilate_radius").default_value(1).scan<'i', int>();
		program.add_argument("--alpha_value").default_value(0.0025).scan<'f', float>();

		// Fitting
		program.add_argument("--fitting_epsilon").default_value(0.001f).scan<'f', float>();
		program.add_argument("--fallback_ransac").implicit_value(true).default_value(false);
		program.add_argument("--num_fitting_points").default_value(10000).scan<'i', int>();

		// Expanding
		program.add_argument("--restricted").implicit_value(true).default_value(false);
		program.add_argument("--max_num_points_refit").default_value(-1).scan<'i', int>();

		// Curve intersection
		program.add_argument("--remove_voronoi_curves").default_value(false).implicit_value(true);
		program.add_argument("--common_points_threshold").default_value(0.02).scan<'f', float>();
		program.add_argument("--shape_epsilon").default_value(0.02).scan<'f', float>();
		program.add_argument("--vertex_threshold").default_value(0.01).scan<'f', float>();

		// Exporting
		program.add_argument("--output_inlier_epsilon").default_value(0.02).scan<'f', float>();
		program.add_argument("--output_alpha_value").default_value(0.0009).scan<'f', float>();
		program.add_argument("--check_curve_before_adding").default_value(false).implicit_value(true);

		program.parse_args(argc, argv);
	}

	fs::path root(program.get<std::string>("data_root"));
	std::string prefix(program.get<std::string>("prefix"));

	if (prefix.empty())
	{
		std::vector<std::string> tasks;
		for (fs::directory_iterator cur_it(root); cur_it != fs::directory_iterator(); ++cur_it)
		{
			// Check if it is end with "feat"
			const auto str = cur_it->path().filename().string();
			if (str.substr(str.size() - 9) != "_feat.npy")
				continue;
			tasks.push_back(cur_it->path().filename().stem().string().substr(0, 8));
		}

		tbb::parallel_for(tbb::blocked_range<int>(0, tasks.size()), [&](const auto& r0)
		{
			for (int i_task = r0.begin(); i_task != r0.end(); ++i_task)
			{
				LOG(INFO) << "Start " << tasks[i_task];

				try
				{
					process_item(
						tasks[i_task],
						root,
						program
					);
				}
				catch (const std::exception& e)
				{
					LOG(INFO) << tasks[i_task] << " failed";
					LOG(INFO) << e.what();
				}
			}
		});
	}
	else
	{
		process_item(
			prefix,
			root,
			program
		);
	}

	return 0;
}
