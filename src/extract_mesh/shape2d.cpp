#include "shape2d.h"


BOOST_CLASS_EXPORT(Shape1D)
BOOST_CLASS_EXPORT(Shape2D)
BOOST_CLASS_EXPORT(MyLine)
BOOST_CLASS_EXPORT(MyEllipse)
BOOST_CLASS_EXPORT(MyCircle)

void MyCircle::find_boundary()
{
    boundary_points.clear();
    std::vector<double> ts(inliers.size());
    std::vector<Eigen::Vector2d> m_parametrics(inliers.size());

    bins.clear();
    bins.resize(num_bins);

    // #pragma omp parallel for
    for (int i = 0; i < inliers.size(); ++i)
    {
        ts[i] = get_parametric(inliers[i]);
        if (ts[i] < 0)
            ts[i] += 2 * M_PI;
        else if(ts[i] > 2 * M_PI)
			ts[i] -= 2 * M_PI;
        int id_bin = ts[i] / bin_range;
        if (id_bin >= num_bins || id_bin < 0)
        {
            LOG(INFO) << id_bin;
            throw;
        }

        bins[id_bin].push_back(ts[i]);

        m_parametrics[i] = Eigen::Vector2d(std::cos(ts[i]), std::sin(ts[i]));
    }

    if (!bins[0].empty() && !bins[num_bins-1].empty())
    {
        bool is_closed = true;

        int id_min_bin=-1, id_max_bin=-1;
	    for(int i=0; i< num_bins;++i)
	    {
            if (bins[i].empty())
                is_closed = false;

		    if (bins[i].empty())
		    {
                if (id_max_bin == -1)
                    id_max_bin = i - 1;
		    }
            else
            {
                if (id_max_bin != -1)
                {
                    id_min_bin = i;
                    break;
                }
            }
	    }

        if (is_closed)
        {
            min_t = 0;
			max_t = 2 * M_PI;
        }
        else
        {
            min_t = *std::min_element(bins[id_min_bin].begin(), bins[id_min_bin].end());
            max_t = *std::max_element(bins[id_max_bin].begin(), bins[id_max_bin].end()) + 2 * M_PI;
        }
    }
    else
    {
        min_t = *std::min_element(ts.begin(), ts.end());
        max_t = *std::max_element(ts.begin(), ts.end());
    }

    boundary_points.emplace_back(get_cartesian(min_t));
    boundary_points.emplace_back(get_cartesian(max_t));
}

Point_set MyCircle::sample_parametric(const int num_samples) const
{
    std::mt19937 rng;

    int num_valid_bins = 0;
    for (const auto& item : bins)
        if (!item.empty())
            num_valid_bins++;

    const int num_sample_per_bin = std::max(1, num_samples / num_valid_bins);

    Point_set point_set;
    for(int i_bin=0;i_bin<num_bins;++i_bin)
    {
        if (bins[i_bin].empty())
			continue;
        std::uniform_real_distribution<double> unif(bin_range * i_bin, bin_range * i_bin + bin_range);
        for (int i = 0; i < num_sample_per_bin; ++i)
        {
            const double t = unif(rng);
            point_set.insert(eigen_2_cgal_point(get_cartesian(t)));
        }
    }
    return point_set;
}

void MyEllipse::find_boundary()
{
    boundary_points.clear();
    std::vector<double> ts(inliers.size());
    std::vector<Eigen::Vector2d> m_parametrics(inliers.size());

    bins.clear();
    bins.resize(num_bins);

    // #pragma omp parallel for
    for (int i = 0; i < inliers.size(); ++i)
    {
        ts[i] = get_parametric(inliers[i]);
        if (ts[i] < 0)
            ts[i] += 2 * M_PI;
        else if (ts[i] > 2 * M_PI)
            ts[i] -= 2 * M_PI;
        int id_bin = ts[i] / bin_range;
        if (id_bin >= num_bins || id_bin < 0)
        {
            LOG(INFO) << id_bin;
            throw;
        }

        bins[id_bin].push_back(ts[i]);

        m_parametrics[i] = Eigen::Vector2d(std::cos(ts[i]), std::sin(ts[i]));
    }

    if (!bins[0].empty() && !bins[num_bins - 1].empty())
    {
        bool is_closed = true;

        int id_min_bin = -1, id_max_bin = -1;
        for (int i = 0; i < num_bins; ++i)
        {
            if (bins[i].empty())
                is_closed = false;

            if (bins[i].empty())
            {
                if (id_max_bin == -1)
                    id_max_bin = i - 1;
            }
            else
            {
                if (id_max_bin != -1)
                {
                    id_min_bin = i;
                    break;
                }
            }
        }

        if (is_closed)
        {
            min_t = 0;
            max_t = 2 * M_PI;
        }
        else
        {
            min_t = *std::min_element(bins[id_min_bin].begin(), bins[id_min_bin].end());
            max_t = *std::max_element(bins[id_max_bin].begin(), bins[id_max_bin].end()) + 2 * M_PI;
        }
    }
    else
    {
        min_t = *std::min_element(ts.begin(), ts.end());
        max_t = *std::max_element(ts.begin(), ts.end());
    }

    boundary_points.emplace_back(get_cartesian(min_t));
    boundary_points.emplace_back(get_cartesian(max_t));
}

Point_set MyEllipse::sample_parametric(const int num_samples) const
{
    std::mt19937 rng;

    int num_valid_bins = 0;
    for (const auto& item : bins)
        if (!item.empty())
            num_valid_bins++;

    const int num_sample_per_bin = std::max(1.,num_samples / (num_valid_bins+1e-6));

    Point_set point_set;

    if (!bins[0].empty() && !bins[num_bins - 1].empty() || true)
    {
        for (int i_bin = 0; i_bin < num_bins; ++i_bin)
        {
            if (bins[i_bin].empty())
                continue;
            std::uniform_real_distribution<double> unif(bin_range * i_bin, bin_range * i_bin + bin_range);
            for (int i = 0; i < num_sample_per_bin; ++i)
            {
                double t = unif(rng);
                t = std::clamp(t, 0., M_PI * 2);
                point_set.insert(eigen_2_cgal_point(get_cartesian(t)));
            }
        }
    }
    else
    {
        std::uniform_real_distribution<double> unif(min_t, max_t);
        for (int i = 0; i < num_samples; ++i)
        {
            double t = unif(rng);
            point_set.insert(eigen_2_cgal_point(get_cartesian(t)));
        }
    }

    
    return point_set;
}


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

void colorize_output_points(const std::vector<std::shared_ptr<Shape>>& v_shapes)
{
	checkFolder("temp/colorized");
	{
		Point_set resampled_points;
		auto index_map_resampled = resampled_points.add_property_map<int>("index").first;
		const auto color_map = get_color_table_bgr2();
		for (int i = 0; i < v_shapes.size(); ++i)
		{
			Point_set local_points = v_shapes[i]->sample_parametric(10000);
			const auto color = color_map[i % color_map.size()];
			local_points.add_property_map<uchar>("red", color[2]);
			local_points.add_property_map<uchar>("green", color[1]);
			local_points.add_property_map<uchar>("blue", color[0]);
			CGAL::IO::write_point_set("temp/colorized/" + std::to_string(i) + "_" + v_shapes[i]->detail_type + ".ply", local_points);
			for (int j = 0; j < local_points.size(); ++j)
			{
				resampled_points.insert(local_points.point(j));
				index_map_resampled[resampled_points.size() - 1] = i;
			}
		}
		colorize_point_set(resampled_points);
		CGAL::IO::write_point_set("temp/colorized/0segmented_resampled.ply", resampled_points);
	}
}
