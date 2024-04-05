#include "writer.h"

#include <npy.hpp>
#include <hdf5.h>

#include "lzf_filter.h"
#include <H5Cpp.h>

void NPYWritter::store_data()
{
	while (true)
	{
		while (!m_queues.empty())
		{
			int id_component;
			std::string prefix;
			std::shared_ptr<unsigned int[]> flag_reshaped;
			std::shared_ptr<unsigned short[]> feature_reshape;
			std::shared_ptr<unsigned short[]> point_feature_reshaped;
			std::shared_ptr<unsigned short[]> sampled_points;
			m_mutex.lock();
			std::tie(id_component, prefix, flag_reshaped, feature_reshape, point_feature_reshaped,
				sampled_points) = m_queues.front();
			m_queues.pop();
			m_mutex.unlock();
			prefix = prefix + "_" + std::to_string(id_component);

			npy::SaveArrayAsNumpy((m_output_root / "training" / (prefix + "_flag.npy")).string(), false, 3,
				std::vector<unsigned long>{resolution, resolution, resolution}.data(),
				flag_reshaped.get());
			npy::SaveArrayAsNumpy((m_output_root / "training" / (prefix + "_pfeat.npy")).string(), false, 4,
				std::vector<unsigned long>{resolution, resolution, resolution, 5}.data(),
				point_feature_reshaped.get());
			npy::SaveArrayAsNumpy((m_output_root / "training" / (prefix + "_feat.npy")).string(), false, 4,
				std::vector<unsigned long>{resolution, resolution, resolution, 3}.data(),
				feature_reshape.get());
			npy::SaveArrayAsNumpy((m_output_root / "training" / (prefix + "_points.npy")).string(), false, 2,
				std::vector<unsigned long>{10000, 6}.data(),
				sampled_points.get());

			// delete[] flag_reshaped;
			// delete[] feature_reshape;
			// delete[] point_feature_reshaped;
			// delete[] sampled_points;
		}
		override_sleep(1);
		if (need_terminal)
			break;
	}
}

H5Writter::H5Writter(const fs::path& m_output_root, const int v_chunk_size, const bool is_udf_feature,
	const bool is_poisson, const bool is_point_feature): Writter(m_output_root, 0), is_udf_feature(is_udf_feature), is_poisson(is_poisson), is_point_feature(is_point_feature), chunk_size(v_chunk_size)
{
	using namespace H5;
	int r = register_lzf();
	std::cout << r << std::endl;
	if (fs::exists((m_output_root / "training.h5").string()) && false)
	{
		LOG(INFO) << "Found existing dataset";
		H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);
		DataSet names = file.openDataSet("names");
		// Read data
		hsize_t dims[1];
		names.getSpace().getSimpleExtentDims(dims);
		file.close();

		cur_dim = dims[0];
		remain_dim = 0;
		extend();
	}
	else
	{
		cur_dim = 0;
		remain_dim = block_dim;
		H5File file((m_output_root / "training.h5").string(), H5F_ACC_TRUNC);
		if (is_udf_feature)
		{
			hsize_t      dims[5] = { block_dim, 256, 256, 256, 5 };
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
			hsize_t      dims[4] = { block_dim, res, res, res };
			hsize_t      maxdims[4] = { H5S_UNLIMITED, res, res, res };
			DataSpace mspace(4, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[4] = { 1, chunk_size, chunk_size, chunk_size };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(4, chunk_dims);
			file.createDataSet("flags",
			                   PredType::NATIVE_UINT32, mspace, cparms);
		}
		if (is_poisson)
		{
			hsize_t      dims[3] = { block_dim, 10000, 6 };
			hsize_t      maxdims[3] = { H5S_UNLIMITED, 10000, 6 };
			DataSpace mspace(3, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[3] = { 1, 10000, 6 };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(3, chunk_dims);
			file.createDataSet("poisson_points",
			                   PredType::NATIVE_UINT16, mspace, cparms);
		}
		if (is_point_feature)
		{
			hsize_t      dims[5] = { block_dim, 256, 256, 256, 5 };
			hsize_t      maxdims[5] = { H5S_UNLIMITED, 256, 256, 256, 5 };
			DataSpace mspace(5, dims, maxdims);

			DSetCreatPropList cparms;
			hsize_t      chunk_dims[5] = { 1, chunk_size, chunk_size, chunk_size, 5 };
			cparms.setShuffle();
			cparms.setFilter(H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL);
			cparms.setChunk(5, chunk_dims);
			file.createDataSet("point_features",
			                   PredType::NATIVE_UINT16, mspace, cparms);
		}
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
}

void H5Writter::extend()
{
	using namespace H5;
	H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);

	if (is_udf_feature)
	{
		DataSet feature_dataset = file.openDataSet("features");
		hsize_t      add_size1[5] = { cur_dim + block_dim, res, res, res, 5 };
		feature_dataset.extend(add_size1);
	}
	if (is_poisson)
	{
		DataSet poisson_dataset = file.openDataSet("poisson_points");
		hsize_t      add_size4[3] = { cur_dim + block_dim, 10000, 6 };
		poisson_dataset.extend(add_size4);
	}
	if(is_point_feature)
	{
		DataSet point_feature_dataset = file.openDataSet("point_features");
		hsize_t      add_size2[5] = { cur_dim + block_dim, res, res, res, 5 };
		point_feature_dataset.extend(add_size2);
	}
	DataSet flag_dataset = file.openDataSet("flags");
	hsize_t      add_size3[4] = { cur_dim + block_dim, res, res, res };
	flag_dataset.extend(add_size3);
	DataSet name_dataset = file.openDataSet("names");
	hsize_t      add_size5[1] = { cur_dim + block_dim };
	name_dataset.extend(add_size5);
	DataSet ids_dataset = file.openDataSet("ids");
	ids_dataset.extend(add_size5);

	remain_dim += block_dim;
	file.close();
}

void H5Writter::store_data()
{
	using namespace H5;
	while (true)
	{
		while (!m_queues.empty())
		{
			auto timer = recordTime();
			std::vector<unsigned int> prefixes;
			std::vector<unsigned int> ids;

			std::shared_ptr<unsigned int[]> flags;
			std::shared_ptr<unsigned short[]> features;
			std::shared_ptr<unsigned short[]> point_features;
			std::shared_ptr<unsigned short[]> sampled_points;
			hsize_t num_items = 0;
			{
				m_mutex.lock();
				num_items = std::min(remain_dim, (int)m_queues.size());
				prefixes.resize(num_items);
				ids.resize(num_items);

				flags = std::shared_ptr<unsigned int[]>(new unsigned int[num_items * res * res * res]);
				if(is_udf_feature)
					features=std::shared_ptr<unsigned short[]>(new unsigned short[num_items * res * res * res * 5]);
				if(is_poisson)
					sampled_points= std::shared_ptr<unsigned short[]>(new unsigned short[num_items * 10000 * 6]);
				if(is_point_feature)
					point_features= std::shared_ptr<unsigned short[]>(new unsigned short[num_items * res * res * res * 5]);


				std::vector<int> id_component(num_items);
				std::vector<std::string> prefix(num_items);
				std::vector<std::shared_ptr<unsigned int[]>> flag(num_items);
				std::vector<std::shared_ptr<unsigned short[]>> feature(num_items);
				std::vector<std::shared_ptr<unsigned short[]>> point_feature(num_items);
				std::vector<std::shared_ptr<unsigned short[]>> sampled_point(num_items);

				for (hsize_t i = 0; i < num_items; ++i)
				{
					std::tie(id_component[i], prefix[i], flag[i], feature[i], point_feature[i],
					         sampled_point[i]) = m_queues.front();

					prefixes[i] = std::atoi(prefix[i].c_str());
					ids[i] = id_component[i];

					m_queues.pop();
				}
				m_mutex.unlock();

#pragma omp parallel for
				for (int i = 0; i < num_items; ++i)
				{
					if (is_udf_feature)
					{
						std::copy_n(feature[i].get(), res * res * res * 5,
						            features.get() + i * res * res * res * 5);
					}
					if (is_poisson)
					{
						std::copy_n(sampled_point[i].get(), 10000 * 6,
						            sampled_points.get() + i * 10000 * 6);
					}
					if (is_point_feature)
					{
						std::copy_n(point_feature[i].get(), res * res * res * 5,
						            point_features.get() + i * res * res * res * 5);
					}
					std::copy_n(flag[i].get(), res * res * res, flags.get() + i * res * res * res);
				}
			}
			profileTime(timer, "Load", false);

			H5File file((m_output_root / "training.h5").string(), H5F_ACC_RDWR);
			if (is_udf_feature)
			{
				DataSet feature_dataset = file.openDataSet("features");
				DataSpace fspace = feature_dataset.getSpace();
				hsize_t     offset[5] = { cur_dim, 0, 0, 0, 0 };
				hsize_t     dims[5] = { num_items, res, res, res, 5 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(5, dims);
				feature_dataset.write(features.get(),
				                      PredType::NATIVE_UINT16, mspace2, fspace);
			}
			if (is_poisson)
			{
				DataSet poisson_dataset = file.openDataSet("poisson_points");
				DataSpace fspace = poisson_dataset.getSpace();
				hsize_t     offset[3] = { cur_dim, 0, 0 };
				hsize_t     dims[3] = { num_items, 10000, 6 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(3, dims);
				poisson_dataset.write(sampled_points.get(),
				                      PredType::NATIVE_UINT16, mspace2, fspace);
			}
			if (is_point_feature)
			{
				DataSet point_feature_dataset = file.openDataSet("point_features");
				DataSpace fspace = point_feature_dataset.getSpace();
				hsize_t     offset[5] = { cur_dim, 0, 0, 0, 0 };
				hsize_t     dims[5] = { num_items, res, res, res, 5 };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(5, dims);
				point_feature_dataset.write(point_features.get(),
				                            PredType::NATIVE_UINT16, mspace2, fspace);
			}
			{
				DataSet flag_dataset = file.openDataSet("flags");
				DataSpace fspace = flag_dataset.getSpace();
				hsize_t     offset[4] = { cur_dim, 0, 0, 0 };
				hsize_t     dims[4] = { num_items, res, res, res };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(4, dims);
				flag_dataset.write(flags.get(),
				                   PredType::NATIVE_UINT32, mspace2, fspace);
			}
			{
				DataSet name_dataset = file.openDataSet("names");
				DataSpace fspace = name_dataset.getSpace();
				hsize_t     offset[1] = { cur_dim };
				hsize_t     dims[1] = { num_items };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(1, dims);
				name_dataset.write(prefixes.data(),
				                   PredType::NATIVE_UINT, mspace2, fspace);
			}
			{
				DataSet name_dataset = file.openDataSet("ids");
				DataSpace fspace = name_dataset.getSpace();
				hsize_t     offset[1] = { cur_dim };
				hsize_t     dims[1] = { num_items };
				fspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
				DataSpace mspace2(1, dims);
				name_dataset.write(ids.data(),
				                   PredType::NATIVE_UINT, mspace2, fspace);
			}
			profileTime(timer, "Write", false);
			file.close();

			flags.reset();
			point_features.reset();
			features.reset();
			point_features.reset();

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

		if (is_udf_feature)
		{
			DataSet feature_dataset = file.openDataSet("features");
			hsize_t      add_size1[5] = { cur_dim, res, res, res, 5 };
			feature_dataset.extend(add_size1);
		}
		if (is_poisson)
		{
			DataSet poisson_dataset = file.openDataSet("poisson_points");
			hsize_t      add_size4[3] = { cur_dim, 10000, 6 };
			poisson_dataset.extend(add_size4);
		}
		if (is_point_feature)
		{
			DataSet point_feature_dataset = file.openDataSet("point_features");
			hsize_t      add_size2[5] = { cur_dim, res, res, res, 5 };
			point_feature_dataset.extend(add_size2);
		}

		DataSet flag_dataset = file.openDataSet("flags");
		hsize_t      add_size3[4] = { cur_dim, res, res, res };
		flag_dataset.extend(add_size3);

		DataSet name_dataset = file.openDataSet("names");
		hsize_t      add_size5[1] = { cur_dim };
		name_dataset.extend(add_size5);

		DataSet ids_dataset = file.openDataSet("ids");
		ids_dataset.extend(add_size5);

		file.close();
	}
}

