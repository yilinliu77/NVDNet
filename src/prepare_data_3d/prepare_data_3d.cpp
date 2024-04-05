#include "prepare_data_3d_parallel.h"

#include "tbb/tbb.h"

#include "omp.h"
#define omp_get_thread_num() 0


std::vector<fs::path> generate_task_list(const fs::path& v_data_input, const fs::path& v_data_output,
                                         const std::string& v_target_list,
                                         const int id_start, const int id_end)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;
	if (v_target_list != "")
	{
		std::ifstream ifs(v_target_list);
		std::string line;
		while (!ifs.eof())
		{
			std::getline(ifs, line);
			if (line.size() < 3)
				break;
			// line = split_string(line, std::string("_"))[0];
			target_prefix.push_back(line);
		}
		ifs.close();
		LOG(INFO) << "Use " << v_target_list << ", we have " << target_prefix.size() << " targets.";
	}
	else
	{
		for (fs::directory_iterator it_file(v_data_input); it_file != fs::directory_iterator(); ++it_file)
		{
			target_prefix.push_back(it_file->path().filename().string());
		}
		LOG(INFO) << "We have " << target_prefix.size() << " targets in total.";
	}

	std::unordered_set<std::string> target_set(target_prefix.begin(), target_prefix.end());
	target_prefix.clear();
	target_prefix.insert(target_prefix.end(), target_set.begin(), target_set.end());

	//Check existing
	std::vector<unsigned int> existing_ids;
	if (fs::exists((v_data_output / "training.h5").string()))
	{
		// using namespace H5;
		// H5File file((v_data_output / "training.h5").string(), H5F_ACC_RDONLY);
		// DataSet dataset = file.openDataSet("names");
		// hsize_t dims[1];
		// dataset.getSpace().getSimpleExtentDims(dims);
		// existing_ids.resize(dims[0]);
		// dataset.read(existing_ids.data(), PredType::NATIVE_UINT);
		// file.close();
	}

#pragma omp parallel for
	for (int i = 0; i < target_prefix.size(); ++i)
	{
		std::string prefix = target_prefix[i];
		const int file_idx = std::atoi(prefix.c_str());
		if (file_idx >= id_start && file_idx < id_end)
		{
			const int file_count = std::count_if(
				boost::filesystem::directory_iterator(v_data_input / prefix),
				boost::filesystem::directory_iterator(),
				static_cast<bool(*)(const boost::filesystem::path&)>(boost::filesystem::is_regular_file)
			);

			if (std::find(existing_ids.begin(), existing_ids.end(), std::atoi(prefix.c_str()))!=existing_ids.end())
				continue;

			const auto feat = v_data_output / "training" / (prefix + "_0_feat.npy");
			const auto pfeat = v_data_output / "training" / (prefix + "_0_pfeat.npy");
			const auto flag = v_data_output / "training" / (prefix + "_0_flag.npy");
			const auto points = v_data_output / "training" / (prefix + "_0_points.npy");

			if (fs::exists(feat) && fs::exists(flag) && fs::exists(points) && fs::exists(pfeat))
				continue;

			// if (prefix != "00000822")
			// continue;

			if (file_count > 2)
			{
#pragma omp critical
				{
					task_files.push_back(v_data_input / prefix);
				}
			}
		}
	}

	std::sort(task_files.begin(), task_files.end(), [](const fs::path& item1, const fs::path& item2)
	{
		return std::atoi(item1.filename().string().c_str()) < std::atoi(item2.filename().string().c_str());
	});
	return task_files;
}

std::vector<Eigen::Vector3i> construct_source_coords(const int v_resolution)
{
	std::vector<Eigen::Vector3i> source_coords(v_resolution * v_resolution * v_resolution);
	int idx = 0;
	for (int x = 0; x < v_resolution; ++x)
		for (int y = 0; y < v_resolution; ++y)
			for (int z = 0; z < v_resolution; ++z)
				source_coords[idx++] = Eigen::Vector3i(x, y, z);
	return source_coords;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<bool>>> construct_target_coords(
	const std::vector<Eigen::Vector3i>& source_coords, int v_resolution)
{
	std::vector<std::vector<int>> target_coords_list;
	std::vector<std::vector<bool>> is_valid_flags;

	for (int dx = -1; dx <= 1; ++dx)
		for (int dy = -1; dy <= 1; ++dy)
			for (int dz = -1; dz <= 1; ++dz)
				if (dx != 0 || dy != 0 || dz != 0)
				{
					std::vector<int> local_coords;
					std::vector<bool> local_is_valid_flags;
					for (const auto& coord : source_coords)
					{
						int new_x = coord[0] + dx;
						int new_y = coord[1] + dy;
						int new_z = coord[2] + dz;
						if (new_x >= 0 && new_x < v_resolution &&
							new_y >= 0 && new_y < v_resolution &&
							new_z >= 0 && new_z < v_resolution)
							local_is_valid_flags.emplace_back(true);
						else
							local_is_valid_flags.emplace_back(false);

						local_coords.emplace_back(new_x * v_resolution * v_resolution + new_y * v_resolution + new_z);
					}
					target_coords_list.emplace_back(local_coords);
					is_valid_flags.emplace_back(local_is_valid_flags);
				}
	return {target_coords_list, is_valid_flags};
}

std::tuple<
	std::vector<Eigen::Vector3i>,
	std::vector<std::vector<int>>,
	std::vector<std::vector<bool>>> construct_graph_3d(const int v_resolution)
{
	auto source_coords = construct_source_coords(v_resolution);
	std::vector<std::vector<int>> target_coords;
	std::vector<std::vector<bool>> valid_flag;
	std::tie(target_coords, valid_flag) = construct_target_coords(source_coords, v_resolution);
	return {source_coords, target_coords, valid_flag};
}



int main(int argc, char* argv[])
{
	oneapi::tbb::global_control global_limit(tbb::global_control::thread_stack_size, 64 * 1024 * 1024);

	argparse::ArgumentParser program("prepare_data_3d");
	{
		LOG(INFO) << "enter the arguments: data_root output_root num_cpus resolution id_start id_end is_log";
		program.add_description("data_root output_root resolution id_start id_end is_log");
		program.add_argument("data_root").required();
		program.add_argument("output_root").required();
		program.add_argument("num_cpus").required().scan<'i', int>();
		program.add_argument("resolution").required().scan<'i', int>();
		program.add_argument("id_start").required().scan<'i', int>();
		program.add_argument("id_end").required().scan<'i', int>();
		program.add_argument("is_log").required().scan<'i', int>();
		program.add_argument("--target_list").default_value(std::string(""));
		program.add_argument("--is_udf_feature").default_value(false).implicit_value(true);
		program.add_argument("--is_poisson").default_value(false).implicit_value(true);
		program.add_argument("--is_point_feature").default_value(false).implicit_value(true);
		program.add_argument("--chunk_size").default_value(32).scan<'i', int>();
		program.parse_args(argc, argv);
	}

	fs::path data_root(program.get<std::string>("data_root"));
	fs::path output_root(program.get<std::string>("output_root"));
	const unsigned long resolution = program.get<int>("resolution");
	const int id_start = program.get<int>("id_start");
	const int id_end = program.get<int>("id_end");
	const bool is_log = program.get<int>("is_log");
	int num_cpus = program.get<int>("num_cpus");
	std::string target_list = program.get<std::string>("--target_list");
	int chunk_size = program.get<int>("--chunk_size");
	const bool is_udf_feature = program.get<bool>("--is_udf_feature");
	const bool is_poisson = program.get<bool>("--is_poisson");
	const bool is_point_feature = program.get<bool>("--is_point_feature");

	assert(id_end >= id_start);

	safeCheckFolder(output_root);
	// safeCheckFolder(output_root / "pointcloud");

	std::vector<Eigen::Vector3i> source_coords;
	std::vector<std::vector<int>> target_coords;
	std::vector<std::vector<bool>> valid_flag;

	auto task_files = generate_task_list(
		data_root,
		output_root,
		target_list,
		id_start,
		id_end
	);

	std::tie(source_coords, target_coords, valid_flag) = construct_graph_3d(resolution);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	LOG(INFO) << ffmt("We have %d CPUs and %d GPUs") % num_cpus % num_gpus;

	std::atomic<size_t> atomic_progress_counter = 0;

	std::vector<Writter*> writers(1);
	std::vector<std::thread> writers_threads;
	writers[0] = new H5Writter(output_root, chunk_size, is_udf_feature, is_poisson, is_point_feature);
	writers_threads.emplace_back(&Writter::store_data, &*writers[0]);

	std::mutex gpu_mutex[100];
	const int max_task_per_gpu = 8;

	std::vector<double> time_statics(10, 0);
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0,task_files.size()), 
		Executer(
			task_files,
			&source_coords, &target_coords, &valid_flag,
			is_log, resolution,
			output_root,
			is_udf_feature,
			is_poisson,
			is_point_feature,

			gpu_mutex,
			max_task_per_gpu,
			num_gpus,
			time_statics,
			writers,
			atomic_progress_counter
		)
	);

	LOG_IF(INFO, is_log) << "Waiting for the writer";
	for (size_t i = 0; i < writers.size(); i++)
	{
		writers[i]->need_terminal = true;
		writers_threads[i].join();
	}
	LOG(INFO) << "Done";

	return 0;
}
