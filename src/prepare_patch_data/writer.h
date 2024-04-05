#pragma once
#include "common_util.h"

#include <hdf5.h>

#include "lzf_filter.h"
#include <H5Cpp.h>

class H5Writter
{
	fs::path m_output_root;

	hsize_t chunk_size = 32;
	hsize_t num_grids = 1200;
	hsize_t num_max_points = 1024;

	hsize_t cur_dim;
	int remain_dim;
	const hsize_t block_dim = 50; // max cache size

public:
	std::mutex m_mutex;
	bool need_terminal = false;
	std::queue<std::tuple<
		int,
		int,
		std::shared_ptr<char[]>,
		std::shared_ptr<short[]>,
		std::shared_ptr<char[]>
		>> m_queues;

	H5Writter(const fs::path& m_output_root, const int v_chunk_size,
		const int v_num_grids
	);

	void extend();

	void store_data();
};

H5Writter::H5Writter(const fs::path& m_output_root, const int v_chunk_size, const int v_num_grids
):m_output_root(m_output_root), chunk_size(v_chunk_size), num_grids(v_num_grids)
{
	using namespace H5;
	int r = register_lzf();
	std::cout << "Initial lzf filter: " << r << std::endl;

	cur_dim = 0;
	remain_dim = block_dim;
	if (chunk_size * chunk_size * chunk_size % 8 != 0)
	{
		LOG(INFO) << "chunk size must be multiple of 8";
		throw std::runtime_error("chunk size must be multiple of 8");
	}

	H5File file((m_output_root / "training.h5").string(), H5F_ACC_TRUNC);
	// voronoi_flags
	{
		hsize_t actual_size = chunk_size * chunk_size * chunk_size / 8;
		hsize_t      dims[3] = { block_dim, num_grids, actual_size };
		hsize_t      maxdims[3] = { H5S_UNLIMITED, num_grids, actual_size };
		DataSpace mspace(3, dims, maxdims);

		DSetCreatPropList cparms;
		hsize_t      chunk_dims[3] = { 1, 1, actual_size };
		cparms.setShuffle();
		cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
		cparms.setChunk(3, chunk_dims);
		file.createDataSet("voronoi_flags",
			PredType::NATIVE_CHAR, mspace, cparms);
	}
	// points
	{
		hsize_t      dims[4] = { block_dim, num_grids, num_max_points, 3 };
		hsize_t      maxdims[4] = { H5S_UNLIMITED, num_grids, num_max_points, 3 };
		DataSpace mspace(4, dims, maxdims);

		DSetCreatPropList cparms;
		hsize_t      chunk_dims[4] = { 1, 1, num_max_points, 3 };
		cparms.setShuffle();
		cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
		cparms.setChunk(4, chunk_dims);
		file.createDataSet("points",
			PredType::NATIVE_SHORT, mspace, cparms);
	}
	// point_flags
	{
		hsize_t      dims[3] = { block_dim, num_grids, num_max_points / 8 };
		hsize_t      maxdims[3] = { H5S_UNLIMITED, num_grids, num_max_points / 8 };
		DataSpace mspace(3, dims, maxdims);

		DSetCreatPropList cparms;
		hsize_t      chunk_dims[3] = { 1, 1, num_max_points / 8 };
		cparms.setShuffle();
		cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
		cparms.setChunk(3, chunk_dims);
		file.createDataSet("point_flags",
			PredType::NATIVE_CHAR, mspace, cparms);
	}
	// names and ids
	{
		hsize_t      dims[1] = { block_dim };
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

void H5Writter::extend()
{
	using namespace H5;
	H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);

	// voronoi_flags
	{
		hsize_t actual_size = chunk_size * chunk_size * chunk_size / 8;
		DataSet dataset = file.openDataSet("voronoi_flags");
		hsize_t      add_size[3] = { cur_dim + block_dim, num_grids, actual_size };
		dataset.extend(add_size);
	}

	// points
	{
		DataSet dataset = file.openDataSet("points");
		hsize_t      add_size[4] = { cur_dim + block_dim, num_grids, num_max_points, 3 };
		dataset.extend(add_size);
	}
	// point_flags
	{
		DataSet dataset = file.openDataSet("point_flags");
		hsize_t      add_size[3] = { cur_dim + block_dim, num_grids, num_max_points / 8 };
		dataset.extend(add_size);
	}
	{
		DataSet name_dataset = file.openDataSet("names");
		hsize_t      add_size[1] = { cur_dim + block_dim };
		name_dataset.extend(add_size);
		DataSet ids_dataset = file.openDataSet("ids");
		ids_dataset.extend(add_size);
	}

	remain_dim += block_dim;
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

			std::shared_ptr<char[]> voronoi_flags;
			std::shared_ptr<short[]> points;
			std::shared_ptr<char[]> point_flags;
			hsize_t num_items = 0;
			{
				m_mutex.lock();
				num_items = std::min(remain_dim, (int)m_queues.size());

				prefixes1.resize(num_items);
				prefixes2.resize(num_items);
				voronoi_flags.reset(new char[num_items * num_grids * chunk_size * chunk_size * chunk_size / 8]);
				points.reset(new short[num_items * num_grids * num_max_points * 3]);
				point_flags.reset(new char[num_items * num_grids * num_max_points / 8]);

				std::vector<std::shared_ptr<char[]>> local_voronoi_flags(num_items);
				std::vector<std::shared_ptr<short[]>> local_points(num_items);
				std::vector<std::shared_ptr<char[]>> local_point_flags(num_items);


				for (hsize_t i = 0; i < num_items; ++i)
				{
					std::tie(prefixes1[i], prefixes2[i], local_voronoi_flags[i], local_points[i], local_point_flags[i]) = m_queues.front();
					m_queues.pop();
				}
				m_mutex.unlock();

				// #pragma omp parallel for
				for (int i = 0; i < num_items; ++i)
				{
					std::copy_n(local_voronoi_flags[i].get(), num_grids * chunk_size * chunk_size * chunk_size / 8,
						voronoi_flags.get() + i * num_grids * chunk_size * chunk_size * chunk_size / 8);
					std::copy_n(local_points[i].get(), num_grids * num_max_points * 3,
						points.get() + i * num_grids * num_max_points * 3);
					std::copy_n(local_point_flags[i].get(), num_grids * num_max_points / 8,
						point_flags.get() + i * num_grids * num_max_points / 8);
				}
			}
			profileTime(timer, "Load", false);

			H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);

			{
				DataSet flags_dataset = file.openDataSet("voronoi_flags");
				DataSpace fspace = flags_dataset.getSpace();
				hsize_t     offset[3] = { cur_dim, 0, 0, };
				hsize_t     dims[3] = { num_items, num_grids, chunk_size * chunk_size * chunk_size / 8 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(3, dims);
				flags_dataset.write(voronoi_flags.get(),
					PredType::NATIVE_CHAR, mspace2, fspace);
			}
			{
				DataSet flags_dataset = file.openDataSet("points");
				DataSpace fspace = flags_dataset.getSpace();
				hsize_t     offset[4] = { cur_dim, 0, 0, 0 };
				hsize_t     dims[4] = { num_items, num_grids, num_max_points, 3 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(4, dims);
				flags_dataset.write(points.get(),
					PredType::NATIVE_SHORT, mspace2, fspace);
			}
			{
				DataSet flags_dataset = file.openDataSet("point_flags");
				DataSpace fspace = flags_dataset.getSpace();
				hsize_t     offset[3] = { cur_dim, 0, 0 };
				hsize_t     dims[3] = { num_items, num_grids, num_max_points / 8 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(3, dims);
				flags_dataset.write(point_flags.get(),
					PredType::NATIVE_CHAR, mspace2, fspace);
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
			remain_dim -= num_items;
			if (remain_dim <= 0)
			{
				extend();
				profileTime(timer, "Extend", false);
			}

		}
		if (need_terminal)
			break;
		override_sleep(1);
	}

	{
		H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);

		// voronoi_flags
		{
			DataSet dataset = file.openDataSet("voronoi_flags");
			hsize_t      add_size[3] = { cur_dim, num_grids, chunk_size* chunk_size* chunk_size/8 };
			dataset.extend(add_size);
		}
		// points
		{
			DataSet dataset = file.openDataSet("points");
			hsize_t      add_size[4] = { cur_dim, num_grids, num_max_points, 3 };
			dataset.extend(add_size);
		}
		// point_flags
		{
			DataSet dataset = file.openDataSet("point_flags");
			hsize_t      add_size[4] = { cur_dim, num_grids, num_max_points / 8 };
			dataset.extend(add_size);
		}
		DataSet name_dataset = file.openDataSet("names");
		hsize_t      add_size5[1] = { cur_dim };
		name_dataset.extend(add_size5);

		DataSet ids_dataset = file.openDataSet("ids");
		ids_dataset.extend(add_size5);

		file.close();
	}
}

