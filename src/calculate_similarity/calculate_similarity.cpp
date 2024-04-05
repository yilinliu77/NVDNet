#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include <argparse/argparse.hpp>

#include <unordered_set>

#include "model_tools.h"
#include "kd_tree_helper.h"

inline
void sampling_on_test_mesh(const fs::path& v_path)
{
	safeCheckFolder(v_path / "sample_for_similarity");

	std::vector<fs::path> mesh_files;
	for (fs::directory_iterator it_file(v_path / "mesh"); it_file != fs::directory_iterator(); ++it_file)
		mesh_files.push_back(it_file->path());

#pragma omp parallel for
	for (int i_file = 0; i_file < mesh_files.size(); ++i_file)
	{
		auto file = mesh_files[i_file];
		if (!fs::exists(v_path / "sample_for_similarity" / file.filename()))
		{
			std::vector<Point_3> train_points;
			std::vector<std::vector<std::size_t>> train_faces;
			std::vector<Triangle_3> train_triangles;
			CGAL::IO::read_PLY(file.string(), train_points, train_faces, CGAL::parameters::use_binary_mode(true));
			for (const auto& face : train_faces)
			{
				train_triangles.emplace_back(
					train_points[face[0]],
					train_points[face[1]],
					train_points[face[2]]
				);
			}
			const int num_points_per_m2 = 1000;
			Point_set train_point_set = sample_points_according_density(train_triangles, num_points_per_m2);
			CGAL::IO::write_point_set((v_path / "sample_for_similarity" / file.filename()).string(), train_point_set);
		}
	}

	LOG(INFO) << "Sampling Done.";
}

inline
void sampling_on_train_mesh(const fs::path& v_path)
{
	safeCheckFolder(v_path / "sample_for_similarity");

	std::vector<fs::path> mesh_files;
	for (fs::directory_iterator it_file(v_path / "mesh"); it_file != fs::directory_iterator(); ++it_file)
		mesh_files.push_back(it_file->path());

#pragma omp parallel for
	for (int i_file = 0; i_file < mesh_files.size(); ++i_file)
	{
		auto file = mesh_files[i_file];
		if (!fs::exists(v_path / "sample_for_similarity" / file.filename()))
		{
			std::vector<Point_3> train_points;
			std::vector<std::vector<std::size_t>> train_faces;
			std::vector<Triangle_3> train_triangles;
			CGAL::IO::read_PLY(file.string(), train_points, train_faces, CGAL::parameters::use_binary_mode(true));
			for (const auto& face : train_faces)
			{
				train_triangles.emplace_back(
					train_points[face[0]],
					train_points[face[1]],
					train_points[face[2]]
				);
			}
			const int num_points_per_m2 = 1000;
			Point_set train_point_set = sample_points_according_density(train_triangles, num_points_per_m2);
			CGAL::IO::write_point_set((v_path / "sample_for_similarity" / file.filename()).string(), train_point_set);
		}
	}

	LOG(INFO) << "Sampling Done.";
}


int main(int argc, char* argv[])
{
	// tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
	argparse::ArgumentParser program("calculate_similarity");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("test_root").required();
		program.add_argument("train_root").required();
		program.add_argument("output_root").required();
		program.add_argument("block_size").required().scan<'i', int>();
		program.add_argument("--prefix").default_value(std::string(""));
		program.add_argument("--test_sample").implicit_value(true).default_value(false);
		program.add_argument("--train_sample").implicit_value(true).default_value(false);
		program.parse_args(argc, argv);
	}

	std::string prefix = program.get<std::string>("--prefix");
	const int BLOCK_SIZE = program.get<int>("block_size");
	const bool is_test_sample = program.get<bool>("--test_sample");
	const bool is_train_sample = program.get<bool>("--train_sample");

	fs::path test_root(program.get<std::string>("test_root"));
	fs::path train_root(program.get<std::string>("train_root"));
	fs::path output_root(program.get<std::string>("output_root"));

	safeCheckFolder(output_root);

	if (is_test_sample)
		sampling_on_test_mesh(test_root);
	
	if (is_train_sample)
		sampling_on_train_mesh(train_root);

	std::vector<fs::path> test_files;
	if (prefix.empty())
	{
		for (fs::directory_iterator it_file(test_root / "sample_for_similarity"); it_file != fs::directory_iterator(); ++it_file)
			test_files.push_back(it_file->path());
	}
	else
		test_files.push_back(test_root / "sample_for_similarity" / (prefix + ".ply"));

	std::vector<fs::path> train_files;
	for (fs::directory_iterator it_file(train_root / "sample_for_similarity"); it_file != fs::directory_iterator(); ++it_file)
	{
		int file_idx = std::stoi(it_file->path().stem().string());
		if (file_idx >= 0 && file_idx <= 800000)
			train_files.push_back(it_file->path());
	}

	LOG(INFO) << ffmt("We have %d test files in total") % test_files.size();
	LOG(INFO) << ffmt("We have %d train files in total") % train_files.size();

	std::vector<std::vector<double>> chamfer_distances(test_files.size(), std::vector<double>(train_files.size(), 0.));

	std::vector<std::string> min_results(test_files.size());
	std::vector<std::string> max_results(test_files.size());

	std::vector<std::shared_ptr<my_kd_tree_t>> kd_test_points(test_files.size());
	std::vector<Point_set> test_point_sets(test_files.size());

	#pragma omp parallel for
	for (int i = 0; i < test_files.size(); ++i)
	{
		auto test_file = test_files[i];
		CGAL::IO::read_point_set(test_file.string(), test_point_sets[i]);
		kd_test_points[i] = std::shared_ptr<my_kd_tree_t>(initialize_kd_tree(test_point_sets[i]));
	}
	LOG(INFO) << "Test K-d Tree Build.";

	int num_finish = 0;
	for (int i_block = 0; i_block < train_files.size(); i_block += BLOCK_SIZE)
	{
		LOG(INFO) << ffmt("#BLOCK %d begin") % (i_block / BLOCK_SIZE);
		std::vector<std::shared_ptr<my_kd_tree_t>> kd_train_points(std::min(BLOCK_SIZE, (int)train_files.size() - i_block));
		std::vector<Point_set> train_point_sets(std::min(BLOCK_SIZE, (int)train_files.size() - i_block));

		#pragma omp parallel for
		for (int i_train = i_block; i_train < std::min(i_block + BLOCK_SIZE, (int)train_files.size()); ++i_train)
		{
			auto train_file = train_files[i_train];
			CGAL::IO::read_point_set(train_file.string(), train_point_sets[i_train - i_block]);
			kd_train_points[i_train - i_block] = std::shared_ptr<my_kd_tree_t>(initialize_kd_tree(train_point_sets[i_train - i_block]));
		}

		#pragma omp parallel for
		for (int i_test = 0; i_test < test_files.size(); ++i_test)
		{
			auto test_file = test_files[i_test];
			std::string file_idx = test_file.stem().string();

			Point_set& test_point_set = test_point_sets[i_test];

			for (int i_train = i_block; i_train < std::min(i_block + BLOCK_SIZE, (int)train_files.size()); ++i_train)
			{
				auto train_file = train_files[i_train];

				Point_set& train_point_set = train_point_sets[i_train - i_block];

				// calculate bi-direction chamfer distance

				// test to train
				double test_chamfer_distance = 0.;
				for (int i_p = 0; i_p < test_point_set.size(); ++i_p)
				{
					auto& test_point = test_point_set.point(i_p);
					test_chamfer_distance += std::sqrt(search_k_neighbour(*kd_train_points[i_train - i_block], cgal_2_eigen_point<float>(test_point), 1).second[0]);
				}
				test_chamfer_distance /= test_point_set.size();

				// train to test
				double train_chamfer_distance = 0.;
				for (int i_p = 0; i_p < train_point_set.size(); ++i_p)
				{
					auto& train_point = train_point_set.point(i_p);
					train_chamfer_distance += std::sqrt(search_k_neighbour(*kd_test_points[i_test], cgal_2_eigen_point<float>(train_point), 1).second[0]);
				}
				train_chamfer_distance /= train_point_set.size();

				auto chamfer_distance = test_chamfer_distance + train_chamfer_distance;

				chamfer_distances[i_test][i_train] = chamfer_distance;
			}
			num_finish += 1;
			LOG(INFO) << ffmt("#Test %s done. %d have done.") % file_idx % num_finish;	
		}
		LOG(INFO) << ffmt("#BLOCK %d done.") % (i_block / BLOCK_SIZE);
	}

	std::ofstream ofs((output_root / "similarity.txt").string());
	for (int i = 0; i < test_files.size(); ++i)
	{
		double min_chamfer = std::numeric_limits<double>::max();
		double max_chamfer = std::numeric_limits<double>::min();
		std::string min_chamfer_matched_prefix = "";
		std::string max_chamfer_matched_prefix = "";
		for (int j = 0; j < train_files.size(); ++j)
		{
			if (chamfer_distances[i][j] < min_chamfer)
			{
				min_chamfer = chamfer_distances[i][j];
				min_chamfer_matched_prefix = train_files[j].stem().string();
			}
			if (chamfer_distances[i][j] > max_chamfer)
			{
				max_chamfer = chamfer_distances[i][j];
				max_chamfer_matched_prefix = train_files[j].stem().string();
			}
		}
		ofs << ffmt("%s %s %f %s %f") % test_files[i].stem().string() % min_chamfer_matched_prefix % min_chamfer % max_chamfer_matched_prefix % max_chamfer << std::endl;
	}

	LOG(INFO) << "Save Done.";
	ofs.close();

	return 0;
}
