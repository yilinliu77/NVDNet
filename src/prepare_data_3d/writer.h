#include <H5public.h>

#include "common_util.h"

class Writter
{
public:
	fs::path m_output_root;
	int i_cpu;
	std::queue<std::tuple<
		int,
		std::string,
		std::shared_ptr<unsigned int[]>,
		std::shared_ptr<unsigned short[]>,
		std::shared_ptr<unsigned short[]>,
		std::shared_ptr<unsigned short[]>
	>> m_queues;

	bool need_terminal = false;
	std::mutex m_mutex;

	Writter()
	{
	}

	Writter(const fs::path& m_output_root, const int i_cpu): m_output_root(m_output_root), i_cpu(i_cpu)
	{
	}

	int get_size()
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		return m_queues.size();
	}

	virtual void store_data() = 0;
};

class NPYWritter : public Writter
{
	const unsigned long resolution = 256;

public:
	NPYWritter() : Writter()
	{
	}

	NPYWritter(const fs::path& m_output_root, const int i_cpu) : Writter(m_output_root, i_cpu)
	{
	}

	void store_data();
};

class H5Writter : public Writter
{
	const hsize_t res = 256;
	const hsize_t block_dim = 50;
	hsize_t chunk_size = 32;
	bool is_udf_feature, is_poisson, is_point_feature;

	hsize_t cur_dim;
	int remain_dim;
public:
	H5Writter(const fs::path& m_output_root, const int v_chunk_size,
		const bool is_udf_feature,
		const bool is_poisson,
		const bool is_point_feature
		);

	void extend();

	void store_data();
};
