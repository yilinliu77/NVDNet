#include <cuda_runtime_api.h>
#include <tbb/tbb.h>

#include "cgal_tools.h"
#include "common_util.h"
#include "model_tools.h"

#include <argparse/argparse.hpp>
#include "calculate_distance.h"
#include "kd_tree_helper.h"

#include "writer.h"

std::vector<fs::path> generate_task_list(const fs::path& v_data_input, const int id_start, const int id_end)
{
	std::vector<fs::path> task_files;

	std::vector<std::string> target_prefix;
	
	for (fs::directory_iterator it_file(v_data_input/"mesh"); it_file != fs::directory_iterator(); ++it_file)
		target_prefix.push_back(it_file->path().filename().stem().string());
	LOG(INFO) << "We have " << target_prefix.size() << " targets in total.";

	//Check existing
	std::vector<std::string> existing_ids;
	fs::directory_iterator it_file(v_data_input / "progress");
	fs::directory_iterator it_end;
	for(;it_file!=it_end;++it_file)
	{
		const std::string full_name = it_file->path().filename().string();
		if (full_name.substr(full_name.size() - 2, 2) != ".udf")
			continue;
		existing_ids.push_back(it_file->path().filename().stem().string());
	}

	#pragma omp parallel for
	for (int i = 0; i < target_prefix.size(); ++i)
	{
		std::string prefix = target_prefix[i];
		const int idx = std::atoi(prefix.c_str());

		if (idx < id_start || idx > id_end)
			continue;

		if (std::find(existing_ids.begin(), existing_ids.end(), prefix) != existing_ids.end())
			continue;

#pragma omp critical
		{
			task_files.push_back(v_data_input / prefix);
		}
	}

	std::sort(task_files.begin(), task_files.end(), [](const fs::path& item1, const fs::path& item2)
	{
		return item1.filename().string() < item2.filename().string();
	});
	return task_files;
}

#pragma optimize("",on)
int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("prepare udf data for training; will read the mesh model you specified");
	{
		LOG(INFO) << "enter the arguments: data_root mesh_folder flag_folder output_folder resolution";
		program.add_description("data_root");
		program.add_argument("data_root").required();
		program.add_argument("mesh_folder").required();
		program.add_argument("flag_folder").required();
		program.add_argument("id_start").required().scan<'i', int>();
		program.add_argument("id_end").required().scan<'i', int>();
		program.add_argument("output_folder").required();
		program.add_argument("resolution").required().scan<'i',int>();
		program.add_argument("--only_nearby").default_value(false).implicit_value(true);
		program.parse_args(argc, argv);
	}

	const int chunk_size = 32;
	const bool only_nearby = program.get<bool>("--only_nearby");
	const int id_start = program.get<int>("id_start");
	const int id_end = program.get<int>("id_end");
	fs::path output_root(program.get<std::string>("data_root"));
	fs::path mesh_folder(output_root/program.get<std::string>("mesh_folder"));
	fs::path flag_folder(output_root/program.get<std::string>("flag_folder"));
	fs::path output_folder(output_root/program.get<std::string>("output_folder"));
	const int res = program.get<int>("resolution");
	if(res*res*res%8!=0)
	{
		LOG(ERROR) << "The resolution should be a multiple of 8";
		return 0;
	}

	safeCheckFolder(output_root);
	safeCheckFolder(output_folder);

	auto task_files = generate_task_list(
		output_root,
		id_start,
		id_end
	);

	LOG(INFO) << ffmt("We have %d valid task") % task_files.size();
	if (task_files.empty())
		return 0;

	H5Writter writer(output_folder, chunk_size, only_nearby);
	std::thread writer_thread(&H5Writter::store_data, &writer);

	const int max_task_per_gpu = 8;
	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	std::mutex gpu_mutex[100];
	std::vector<Point_3> query_points(res*res*res);
	for(int i=0;i<res*res*res;++i)
	{
		const int x = i / res / res;
		const int y = (i / res) % res;
		const int z = i % res;
		query_points[i] = Point_3(
			(double)x / (res - 1) * 2 - 1,
			(double)y / (res - 1) * 2 - 1,
			(double)z / (res - 1) * 2 - 1
		);
	}

	std::atomic<int> progress(0);


	tbb::parallel_for(tbb::blocked_range<int>(0, task_files.size()),
		[&](const tbb::blocked_range<int>& range)
		{
			for (int i_task = range.begin(); i_task != range.end(); ++i_task)
			{
				progress.fetch_add(1);
				if (progress.load() % 100 == 0)
					LOG(INFO) << ffmt("Progress: %d/%d") % progress.load() % task_files.size();

				std::string prefix = task_files[i_task].filename().string();
				int prefix1 = std::atoi(prefix.substr(0, prefix.find_first_of('_')).c_str());
				int prefix2 = std::atoi(prefix.substr(prefix.find_first_of('_') + 1).c_str());

				std::vector<Point_3> points;
				std::vector<Triangle_3> triangles;
				std::vector<Vector_3> normals;
				std::vector<std::array<int, 3>> face_indices_array;
				read_model((mesh_folder / (prefix + ".ply")).string(), points, normals, face_indices_array, triangles);
				std::vector<std::vector<int>> face_indices(face_indices_array.size());
				for(int  i=0;i< face_indices_array.size();++i)
				{
					face_indices[i].resize(3);
					for(int j=0;j<3;++j)
						face_indices[i][j] = face_indices_array[i][j];
				}

				std::ifstream ifs((flag_folder / (prefix + ".bin")).string(), std::ios::binary);
				std::vector<char> raw_flags(std::istreambuf_iterator<char>(ifs), {});
				ifs.close();
				if (triangles.size() <= 10 || raw_flags.empty())
					continue;

				std::vector<char> flags(res * res * res, false);
				for (int i = 0; i < raw_flags.size(); ++i)
					for (int j = 0; j < 8; ++j)
						flags[i * 8 + j] = (char)((raw_flags[i] >> j) & 1);

				// 4. Calculate the closest primitive and distance for query points
				Eigen::Tensor<unsigned short, 2, Eigen::RowMajor> udf_feature(query_points.size(), 5);
				{
					try
					{
						bool succesful = false;
						while (!succesful)
						{
							for (int i = 0; i < max_task_per_gpu * num_gpus; ++i)
							{
								if (gpu_mutex[i].try_lock())
								{
									cudaSetDevice(i % num_gpus);
									calculate_distance(
										query_points, points, face_indices,
										false, udf_feature);
									succesful = true;
									gpu_mutex[i].unlock();
									break;
								}
							}
							if (!succesful)
								override_sleep(3);
						}
					}
					catch (...)
					{
						LOG(ERROR) << prefix << " failed";
						exit(0);
					}
				}

				// Write
				auto timer = recordTime();
				{
					while (writer.m_queues.size() > 50)
						override_sleep(1);

					if (only_nearby)
					{
						std::vector<unsigned short> chunked_udf_feature;
						std::vector<char> chunked_flags;
						std::vector<Point_3> chunked_query_points;

						for(int x=0;x<res;x+= chunk_size)
							for (int y = 0; y < res; y += chunk_size)
								for (int z = 0; z < res; z += chunk_size)
								{
									std::vector<unsigned short> feature_data(chunk_size * chunk_size * chunk_size * 5);
									std::vector<char> flag_data(chunk_size * chunk_size * chunk_size);
									std::vector<Point_3> points_data(chunk_size * chunk_size * chunk_size);
									int num_valids = 0;
									for (int i = 0; i < chunk_size; ++i)
										for (int j = 0; j < chunk_size; ++j)
											for (int k = 0; k < chunk_size; ++k)
											{
												const int id_local1 = i * chunk_size * chunk_size * 5 + j * chunk_size * 5 + k * 5 + 0;
												const int id_local2 = i * chunk_size * chunk_size + j * chunk_size + k + 0;
												const int idx = (x + i) * res * res + (y + j) * res + (z + k);
												if ((double)udf_feature(idx,0) / 65535. * 2 < 0.3)
												{
													num_valids += 1;
												}
												feature_data[id_local1 + 0] = udf_feature(idx, 0);
												feature_data[id_local1 + 1] = udf_feature(idx, 1);
												feature_data[id_local1 + 2] = udf_feature(idx, 2);
												feature_data[id_local1 + 3] = udf_feature(idx, 3);
												feature_data[id_local1 + 4] = udf_feature(idx, 4);
												flag_data[id_local2] = flags[idx];
												points_data[id_local2] = query_points[idx];
											}
									if (num_valids / (double)(chunk_size * chunk_size * chunk_size) < 0.5)
										continue;
									chunked_udf_feature.insert(chunked_udf_feature.end(), feature_data.begin(), feature_data.end());
									chunked_flags.insert(chunked_flags.end(), flag_data.begin(), flag_data.end());
									chunked_query_points.insert(chunked_query_points.end(), points_data.begin(), points_data.end());
								}

						std::shared_ptr<unsigned short[]> feature_ptr(new unsigned short[chunked_udf_feature.size()]);
						std::shared_ptr<char[]> flag_ptr(new char[chunked_flags.size()]);

						const int size1 = chunked_udf_feature.size();
						const int size2 = chunked_flags.size();
						std::move(chunked_udf_feature.begin(), chunked_udf_feature.end(), feature_ptr.get());
						std::move(chunked_flags.begin(), chunked_flags.end(), flag_ptr.get());

						writer.m_mutex.lock();
						writer.m_queues.emplace(
							prefix1,
							prefix2,
							feature_ptr,
							flag_ptr,
							size1,
							size2
						);
						writer.m_mutex.unlock();

						if (false)
						{
							Point_set p;
							p.resize(chunked_query_points.size());
							for (int i = 0; i < chunked_query_points.size(); ++i)
							{
								const double udf = chunked_udf_feature[i * 5 + 0] / 65535. * 2;
								const double dphi = chunked_udf_feature[i * 5 + 1] / 65535. * M_PI * 2;
								const double dtheta = chunked_udf_feature[i * 5 + 2] / 65535. * M_PI * 2;
								Vector_3 dir(std::sin(dtheta) * std::cos(dphi), std::sin(dtheta) * std::sin(dphi), std::cos(dtheta));
								p.point(i) = chunked_query_points[i];
								// p.point(i) = chunked_query_points[i] + dir * udf;
							}
							CGAL::IO::write_point_set((output_root/"debug"/ "1.ply").string(), p);
						}
						
					}
					else
					{
						// std::vector<unsigned short> data(query_points.size() * 5);
						// std::move(udf_feature.data(), udf_feature.data() + query_points.size() * 5, data.data());
						// writer.m_mutex.lock();
						// writer.m_queues.emplace(
						// 	prefix1,
						// 	prefix2,
						// 	data,
						// 	flags
						// );
						// writer.m_mutex.unlock();
					}
				}
				profileTime(timer, "Write " + prefix, false);

				std::ofstream out((output_root / "progress" / (ffmt("%d.udf") % prefix).str()).string());
				out << "";
				out.close();
			}
		});

	writer.need_terminal = true;
	writer_thread.join();

	LOG(INFO) << "Done";
	return 0;
}
