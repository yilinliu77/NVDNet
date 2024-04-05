#pragma once
#include "common_util.h"

#include <hdf5.h>

#include "lzf_filter.h"
#include <H5Cpp.h>

class H5Writter
{
	fs::path m_output_root;
	const hsize_t res = 256;
	hsize_t chunk_size = 32;
	bool only_nearby;

	hsize_t cur_dim;
	int remain_dim;
public:
	std::mutex m_mutex;
	bool need_terminal = false;
	hsize_t block_dim;
	std::queue<std::tuple<
		int,
		int,
		std::shared_ptr<unsigned short[]>,
		std::shared_ptr<char[]>,
			int,int
		>> m_queues;

	H5Writter(const fs::path& m_output_root, const int v_chunk_size,
		const bool is_udf_feature
	);

	void extend(const int num_items);

	void store_data();
};

H5Writter::H5Writter(const fs::path& m_output_root, const int v_chunk_size, const bool only_nearby
):m_output_root(m_output_root), only_nearby(only_nearby), chunk_size(v_chunk_size)
{
	using namespace H5;
	int r = register_lzf();
	std::cout << "Initial lzf filter: " << r << std::endl;

	if (only_nearby)
		block_dim = 10000;
	else
		block_dim = 50;

	cur_dim = 0;
	remain_dim = block_dim;
	H5File file((m_output_root / "training.h5").string(), H5F_ACC_TRUNC);
	if(only_nearby)
	{
		{
			hsize_t      dims[5] = { 0, 32, 32, 32, 5 };
			hsize_t      maxdims[5] = { H5S_UNLIMITED, 32, 32, 32, 5 };
			DataSpace mspace(5, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[5] = { 1, chunk_size, chunk_size, chunk_size, 5 };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(5, chunk_dims);
			file.createDataSet("features",
				PredType::NATIVE_UINT16, mspace, cparms);
		}
		{
			hsize_t      dims[4] = { 0, 32, 32, 32 };
			hsize_t      maxdims[4] = { H5S_UNLIMITED, 32, 32, 32 };
			DataSpace mspace(4, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[4] = { 1, chunk_size, chunk_size, chunk_size };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(4, chunk_dims);
			file.createDataSet("flags",
				PredType::NATIVE_CHAR, mspace, cparms);
		}
	}
	else
	{
		{
			hsize_t      dims[5] = { 0, 256, 256, 256, 5 };
			hsize_t      maxdims[5] = { H5S_UNLIMITED, 256, 256, 256, 5 };
			DataSpace mspace(5, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[5] = { 1, chunk_size, chunk_size, chunk_size, 5 };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(5, chunk_dims);
			file.createDataSet("features",
				PredType::NATIVE_UINT16, mspace, cparms);
		}
		{
			hsize_t      dims[4] = { 0, 256, 256, 256 };
			hsize_t      maxdims[4] = { H5S_UNLIMITED, 256, 256, 256 };
			DataSpace mspace(4, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[4] = { 1, chunk_size, chunk_size, chunk_size };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(4, chunk_dims);
			file.createDataSet("flags",
				PredType::NATIVE_CHAR, mspace, cparms);
		}
	}
	
	{
		hsize_t      dims[1] = { 0 };
		hsize_t      maxdims[1] = { H5S_UNLIMITED };
		DataSpace mspace(1, dims, maxdims);
		DSetCreatPropList cparms;
		hsize_t      chunk_dims[1] = { 1 };
		cparms.setChunk(1, chunk_dims);
		file.createDataSet("names",
			PredType::NATIVE_UINT, mspace, cparms);

		mspace = DataSpace(1, dims, maxdims);
		file.createDataSet("ids",
			PredType::NATIVE_UINT, mspace, cparms);
	}
	file.close();
}

void H5Writter::extend(const int num_items)
{
	using namespace H5;
	H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);

	const hsize_t target_dim = cur_dim + num_items;

	if (only_nearby)
	{
		{
			DataSet feature_dataset = file.openDataSet("features");
			hsize_t      add_size1[5] = { target_dim, chunk_size, chunk_size, chunk_size, 5 };
			feature_dataset.extend(add_size1);
		}
		{
			DataSet flag_dataset = file.openDataSet("flags");
			hsize_t      add_size2[4] = { target_dim, chunk_size, chunk_size, chunk_size };
			flag_dataset.extend(add_size2);
		}
	}
	else
	{
		{
			DataSet feature_dataset = file.openDataSet("features");
			hsize_t      add_size1[5] = { target_dim, res, res, res, 5 };
			feature_dataset.extend(add_size1);
		}
		{
			DataSet flag_dataset = file.openDataSet("flags");
			hsize_t      add_size2[4] = { target_dim, res, res, res };
			flag_dataset.extend(add_size2);
		}
	}
	
	DataSet name_dataset = file.openDataSet("names");
	hsize_t      add_size5[1] = { target_dim };
	name_dataset.extend(add_size5);
	DataSet ids_dataset = file.openDataSet("ids");
	ids_dataset.extend(add_size5);
	file.close();
}

// #pragma optimize("", off)
void H5Writter::store_data()
{
	using namespace H5;
	while (true)
	{
		while (!m_queues.empty())
		{
			auto timer = recordTime();
			std::vector<unsigned int> prefixes1;
			std::vector<unsigned int> prefixes2;

			std::vector<unsigned short> features;
			std::vector<char> flags;
			hsize_t num_items = 0;

			if (only_nearby)
			{
				m_mutex.lock();
				const int num_shapes = (int)m_queues.size();

				std::vector<std::shared_ptr<unsigned short[]>> all_feature_ptrs(num_shapes);
				std::vector<std::shared_ptr<char[]>> all_flag_ptrs(num_shapes);
				std::vector<int> num_feature_items(num_shapes);
				std::vector<int> num_flag_items(num_shapes);

				std::vector<int> prefix1(num_shapes);
				std::vector<int> prefix2(num_shapes);


				for (hsize_t i = 0; i < num_shapes; ++i)
				{
					std::tie(prefix1[i], prefix2[i], all_feature_ptrs[i], all_flag_ptrs[i], num_feature_items[i], num_flag_items[i]) = m_queues.front();
					m_queues.pop();
				}
				m_mutex.unlock();

				features.resize(std::accumulate(num_feature_items.begin(), num_feature_items.end(), 0));
				flags.resize(std::accumulate(num_flag_items.begin(), num_flag_items.end(), 0));
				num_items = features.size() / (chunk_size * chunk_size * chunk_size * 5);
				prefixes1.resize(num_items);
				prefixes2.resize(num_items);

				tbb::parallel_for(tbb::blocked_range<int>(0, num_shapes), [&](const auto& r0)
					{
						for (int i = r0.begin(); i < r0.end(); ++i)
						{
							const int id_feature_start = std::accumulate(num_feature_items.begin(), num_feature_items.begin() + i, 0);
							const int id_flag_start = std::accumulate(num_flag_items.begin(), num_flag_items.begin() + i, 0);
							std::copy_n(all_feature_ptrs[i].get(), num_feature_items[i], features.begin() + id_feature_start);
							std::copy_n(all_flag_ptrs[i].get(), num_flag_items[i], flags.begin() + id_flag_start);

							const int num_items_local = num_feature_items[i] / (chunk_size * chunk_size * chunk_size * 5);
							if (num_feature_items[i] % (chunk_size * chunk_size * chunk_size * 5) != 0)
							{
								LOG(ERROR) << "Wrong data";
								throw;
							}

							const int id_shape_start = id_feature_start / (chunk_size * chunk_size * chunk_size * 5);

							std::fill_n(prefixes1.begin() + id_shape_start, num_items_local, prefix1[i]);
							std::fill_n(prefixes2.begin() + id_shape_start, num_items_local, prefix2[i]);
							// prefixes1.insert(prefixes1.end(), num_items_local, prefix1[i]);
							// prefixes2.insert(prefixes2.end(), num_items_local, prefix2[i]);
							// num_items += num_items_local;
						}
					});
				
			}
			else
			{
				m_mutex.lock();
				num_items = std::min(remain_dim, (int)m_queues.size());

				features.resize(num_items * res * res * res * 5);
				flags.resize(num_items * res * res * res);
				prefixes1.resize(num_items);
				prefixes2.resize(num_items);

				std::vector<std::vector<unsigned short>> feature(num_items);
				std::vector<std::vector<char>> flag(num_items);

				for (hsize_t i = 0; i < num_items; ++i)
				{
					throw;
					// std::tie(prefixes1[i], prefixes2[i],feature[i], flag[i]) = m_queues.front();
					// m_queues.pop();
				}
				m_mutex.unlock();

				// #pragma omp parallel for
				for (int i = 0; i < num_items; ++i)
				{
					if (only_nearby)
					{
						features.insert(features.end(), feature[i].begin(), feature[i].end());
						flags.insert(flags.end(), flag[i].begin(), flag[i].end());

						const int num_items_local = features.size() / (chunk_size * chunk_size * chunk_size * 5);
						if (features.size() % (chunk_size * chunk_size * chunk_size * 5) != 0)
						{
							LOG(ERROR) << "Wrong data";
							throw;
						}
						prefixes1.insert(prefixes1.end(),prefixes1[i], num_items_local);
						prefixes2.insert(prefixes2.end(), prefixes2[i], num_items_local);
					}
					else
					{
						std::copy_n(feature[i].begin(), res * res * res * 5,
														features.begin() + i * res * res * res * 5);
						std::copy_n(flag[i].begin(), res * res * res,
														flags.begin() + i * res * res * res);
					}
				}
			}
			profileTime(timer, "Load", false);

			extend(num_items);
			profileTime(timer, "Extend", false);

			H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);
			if (only_nearby)
			{
				{
					DataSet feature_dataset = file.openDataSet("features");
					DataSpace fspace = feature_dataset.getSpace();
					hsize_t     offset[5] = { cur_dim, 0, 0, 0, 0 };
					hsize_t     dims[5] = { num_items, chunk_size, chunk_size, chunk_size, 5 };
					fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
					DataSpace mspace2(5, dims);
					feature_dataset.write(features.data(),
						PredType::NATIVE_UINT16, mspace2, fspace);
				}
				{
					DataSet flags_dataset = file.openDataSet("flags");
					DataSpace fspace = flags_dataset.getSpace();
					hsize_t     offset[4] = { cur_dim, 0, 0, 0 };
					hsize_t     dims[4] = { num_items, chunk_size, chunk_size, chunk_size };
					fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
					DataSpace mspace2(4, dims);
					flags_dataset.write(flags.data(),
						PredType::NATIVE_CHAR, mspace2, fspace);
				}
			}
			else
			{
				{
					DataSet feature_dataset = file.openDataSet("features");
					DataSpace fspace = feature_dataset.getSpace();
					hsize_t     offset[5] = { cur_dim, 0, 0, 0, 0 };
					hsize_t     dims[5] = { num_items, res, res, res, 5 };
					fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
					DataSpace mspace2(5, dims);
					feature_dataset.write(features.data(),
						PredType::NATIVE_UINT16, mspace2, fspace);
				}
				{
					DataSet flags_dataset = file.openDataSet("flags");
					DataSpace fspace = flags_dataset.getSpace();
					hsize_t     offset[4] = { cur_dim, 0, 0, 0 };
					hsize_t     dims[4] = { num_items, res, res, res };
					fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
					DataSpace mspace2(4, dims);
					flags_dataset.write(flags.data(),
						PredType::NATIVE_CHAR, mspace2, fspace);
				}
			}
			{
				DataSet name_dataset = file.openDataSet("names");
				DataSpace fspace = name_dataset.getSpace();
				hsize_t     offset[1] = { cur_dim };
				hsize_t     dims[1] = { num_items };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(1, dims);
				name_dataset.write(prefixes1.data(),
					PredType::NATIVE_UINT, mspace2, fspace);
			}
			{
				DataSet name_dataset = file.openDataSet("ids");
				DataSpace fspace = name_dataset.getSpace();
				hsize_t     offset[1] = { cur_dim };
				hsize_t     dims[1] = { num_items };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(1, dims);
				name_dataset.write(prefixes2.data(),
					PredType::NATIVE_UINT, mspace2, fspace);
			}
			profileTime(timer, "Write", false);
			file.close();

			cur_dim += num_items;
		}
		if (need_terminal)
			break;
		override_sleep(1);
	}
}

