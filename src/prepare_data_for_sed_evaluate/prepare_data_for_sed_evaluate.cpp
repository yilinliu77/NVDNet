#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include <argparse/argparse.hpp>

#include <unordered_set>

#include "model_tools.h"
#include "kd_tree_helper.h"

void colorize_point_set(Point_set& v_points, const std::string& v_name)
{
	const auto index_map = v_points.property_map<int>(v_name).first;
	if (index_map == nullptr)
	{
		LOG(ERROR) << "Cannot find index map";
		return;
	}
	const auto color_map = get_color_table_bgr2();
	auto rmap = v_points.add_property_map<uchar>("red").first;
	auto gmap = v_points.add_property_map<uchar>("green").first;
	auto bmap = v_points.add_property_map<uchar>("blue").first;
	for (int i = 0; i < v_points.size(); ++i)
	{
		rmap[i] = color_map[index_map[i] % color_map.size()][2];
		gmap[i] = color_map[index_map[i] % color_map.size()][1];
		bmap[i] = color_map[index_map[i] % color_map.size()][0];
	}
}


int main(int argc, char* argv[])
{
	// tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("calculate_similarity");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("test_root").required();
		program.add_argument("final_mesh_root").required();
		program.add_argument("output_root").required();
		program.add_argument("--prefix").default_value(std::string(""));
		program.add_argument("--epsilon").default_value(0.0025).scan<'f', float>();
		program.parse_args(argc, argv);
	}

	std::string prefix = program.get<std::string>("--prefix");
	const float epsilon = program.get<float>("--epsilon");

	fs::path test_root(program.get<std::string>("test_root"));
	fs::path final_mesh_root(program.get<std::string>("final_mesh_root"));
	fs::path output_root(program.get<std::string>("output_root"));

	safeCheckFolder(output_root);

	std::vector<fs::path> test_files;
	if (prefix.empty())
	{
		for (fs::directory_iterator it_file(test_root); it_file != fs::directory_iterator(); ++it_file)
			test_files.push_back(it_file->path());
	}
	else
		test_files.push_back(test_root / (prefix + ".ply"));


	LOG(INFO) << ffmt("We have %d test files in total") % test_files.size();

#pragma omp parallel for
	for (int i_test = 0; i_test < test_files.size(); ++i_test)
	{
		fs::path test_file = test_files[i_test];
		std::string file_idx = test_file.stem().string();
		// 1. read point set
		Point_set test_point_set;
		CGAL::IO::read_point_set(test_file.string(), test_point_set);

		// 2. read surfaces and curves with primitive index
		Point_set surfaces, curves;
		CGAL::IO::read_point_set((final_mesh_root / file_idx / "eval" / "surfaces.ply").string(), surfaces);
		CGAL::IO::read_point_set((final_mesh_root / file_idx / "eval" / "curves.ply").string(), curves);

		// 3. split surfaces and curves;
		std::vector<Point_set> surfaces_split, curves_split;
		auto surfaces_prmitive_index = surfaces.property_map<int>("primitive_index").first;
		auto curves_prmitive_index = curves.property_map<int>("primitive_index").first;
		int num_surfaces = 0, num_curves = 0;
		for (int i_p = 0; i_p < surfaces.size(); ++i_p)
			num_surfaces = std::max(num_surfaces, (int)surfaces_prmitive_index[i_p]);
		num_surfaces += 1;

		surfaces_split.resize(num_surfaces);
		for (int i_p = 0; i_p < surfaces.size(); ++i_p)
		{
			int id = (int)surfaces_prmitive_index[i_p];
			surfaces_split[id].insert(surfaces.point(i_p));
		}

		for (int i_p = 0; i_p < curves.size(); ++i_p)
			num_curves = std::max(num_curves, (int)curves_prmitive_index[i_p]);
		num_curves += 1;

		curves_split.resize(num_curves);
		for (int i_p = 0; i_p < curves.size(); ++i_p)
		{
			int id = (int)curves_prmitive_index[i_p];
			curves_split[id].insert(curves.point(i_p));
		}


		LOG(INFO) << ffmt("Num: Curves: %d, Surfaces: %d") % num_curves % num_surfaces;

		// 4. allocate each point to a surface
		std::vector<std::shared_ptr<my_kd_tree_t>> kd_surfaces_split(num_surfaces);

		for (int i_surface = 0; i_surface < num_surfaces; ++i_surface)
			kd_surfaces_split[i_surface] = std::shared_ptr<my_kd_tree_t>(initialize_kd_tree(surfaces_split[i_surface]));

		auto primitive_index = test_point_set.add_property_map<int>("primitive_index", -1).first;
		for (int i_p = 0; i_p < test_point_set.size(); ++i_p)
		{
			auto& point = test_point_set.point(i_p);
			double min_distance = std::numeric_limits<double>::max();
			for (int i_surface = 0; i_surface < num_surfaces; ++i_surface)
			{
				auto& surface = surfaces_split[i_surface];
				double distance = search_k_neighbour(*kd_surfaces_split[i_surface], cgal_2_eigen_point<float>(point), 1).second[0];
				if (distance < min_distance)
				{
					min_distance = distance;
					primitive_index[i_p] = i_surface;
				}
			}
		}

		// 5. find point in the curves
		std::vector<std::shared_ptr<my_kd_tree_t>> kd_curves_split(num_curves);

		for (int i_curve = 0; i_curve < num_curves; ++i_curve)
			kd_curves_split[i_curve] = std::shared_ptr<my_kd_tree_t>(initialize_kd_tree(curves_split[i_curve]));

		auto curve_index = test_point_set.add_property_map<int>("curve_index", -1).first;
		for (int i_p = 0; i_p < test_point_set.size(); ++i_p)
		{
			auto& point = test_point_set.point(i_p);
			double min_distance = std::numeric_limits<double>::max();
			for (int i_curve = 0; i_curve < num_curves; ++i_curve)
			{
				auto& curve = curves_split[i_curve];
				double distance = search_k_neighbour(*kd_curves_split[i_curve], cgal_2_eigen_point<float>(point), 1).second[0];
				if (distance < epsilon && distance < min_distance)
				{
					min_distance = distance;
					curve_index[i_p] = i_curve;
				}
			}
		}

		// 6. output surfaces and curves with id
		Point_set o_curves;
		auto o_primitive_index = o_curves.add_property_map<int>("primitive_index", -1).first;
		for (int i_p = 0; i_p < test_point_set.size(); ++i_p)
		{
			if (curve_index[i_p] != -1)
			{
				auto p = o_curves.insert(test_point_set.point(i_p));
				o_primitive_index[*p] = curve_index[i_p];
			}
		}

		colorize_point_set(test_point_set, "primitive_index");
		colorize_point_set(o_curves, "primitive_index");
		safeCheckFolder(output_root / file_idx / "eval");
		CGAL::IO::write_point_set((output_root / file_idx / "eval" / "surfaces.ply").string(), test_point_set);
		CGAL::IO::write_point_set((output_root / file_idx / "eval" / "curves.ply").string(), o_curves);
	}

	return 0;
}
