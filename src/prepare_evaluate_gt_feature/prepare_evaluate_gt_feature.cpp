#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"

#include <argparse/argparse.hpp>


#include "calculate_distance.h"
#include "TinyNPY.h"

std::vector<fs::path> generate_task_list(const fs::path& v_data_input)
{
	std::vector<fs::path> task_files;

	for (fs::directory_iterator it_file(v_data_input); it_file != fs::directory_iterator(); ++it_file)
		task_files.push_back(it_file->path());
	LOG(INFO) << "We have " << task_files.size() << " targets in total.";

	return task_files;
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root";
		program.add_description("data_root");
		program.add_argument("input_folder").required();
		program.add_argument("output_folder").required();
		program.add_argument("--resolution").default_value(256).scan<'i', int>();
		program.parse_args(argc, argv);
	}

	fs::path data_root(program.get<std::string>("input_folder"));
	fs::path output_root(program.get<std::string>("output_folder"));
	const int resolution = program.get<int>("--resolution");

	safeCheckFolder(output_root);

	auto task_files = generate_task_list(
		data_root
	);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;

	const int max_task_per_gpu = 8;
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	tbb::global_control limit(tbb::global_control::max_allowed_parallelism, num_gpus * max_task_per_gpu);

	std::atomic<int> progress(0);

	// Generate query points
	std::vector<Point_3> total_query_points(resolution * resolution * resolution);
	for (int i = 0; i < total_query_points.size(); ++i)
	{
		const int x = i / resolution / resolution;
		const int y = i / resolution % resolution;
		const int z = i % resolution;
		total_query_points[i] = Point_3(
			(double)x / (resolution - 1) * 2 - 1,
			(double)y / (resolution - 1) * 2 - 1,
			(double)z / (resolution - 1) * 2 - 1
		);
	}

	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()),
		[&](const tbb::blocked_range<int>& r)
		{
			for (int i_task = r.begin(); i_task != r.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().stem().string();
				auto timer_io = recordTime();

				// 1. Read and preprocess the mesh and primitives
				std::vector<Point_3> total_vertices;
				std::vector<std::vector<int>> total_faces;
				{
					CGAL::IO::read_polygon_soup(
						task_files[i_task].string(),
						total_vertices,
						total_faces
					);
				}

				// 4. Compute closest_primitives
				Eigen::Tensor<unsigned short, 2, Eigen::RowMajor> features(resolution * resolution * resolution, 5);
				{
					cudaSetDevice(i_task % num_gpus);
					calculate_distance(
						total_query_points,
						total_vertices,
						total_faces,
						false, features
					);
				}

				std::vector<unsigned short> features_vec(resolution * resolution * resolution * 5);
				std::copy_n(features.data(), features_vec.size(), features_vec.begin());

				NpyArray::SaveNPY(
					(output_root / (prefix + ".npy")).string(),
					features_vec,
					NpyArray::shape_t{ (size_t)resolution, (size_t)resolution, (size_t)resolution, 5 }
				);

				std::ofstream out((output_root / "progress" / (ffmt("%d.0") % prefix).str()).string());
				out << "";
				out.close();
			}
		}
	);
	

	LOG(INFO) << "Done";
	return 0;
}
