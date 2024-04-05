#pragma once
#ifndef COMMON_UTIL
#define COMMON_UTIL

#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#define NOMINMAX

#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <tuple>
#include <string>
#include <queue>
#include <algorithm>

#include <glog/logging.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>
#include "cmake_definition.h" // define eigen plugins here

#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include <json/reader.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = boost::filesystem;
typedef boost::format ffmt;

void checkFolder(const fs::path& folder);
void safeCheckFolder(const fs::path& folder);
Json::Value setup_args(int argc, char** argv, const std::string& v_name);
std::chrono::steady_clock::time_point recordTime();
float getTime(std::chrono::steady_clock::time_point& now);
double profileTime(std::chrono::steady_clock::time_point& now, std::string vTip = "", bool v_profile = true);
void debug_img(const cv::Mat& v_img);
void debug_img(std::vector<cv::Mat>& vImgs);
void wait_interaction();
void override_sleep(float seconds);
std::vector<cv::Vec3b> get_color_table_bgr();
std::vector<cv::Vec3b> get_color_table_bgr2();
std::vector<cv::Vec3b> get_color_table_bgr3();
std::vector<cv::Vec3b> get_color_table_bgr4();

//inline std::mt19937_64 global_random_seed(0);
extern bool interaction_on;

class Global_helper
{
private:
    static std::mt19937_64 m_rng;

public:
    static void seed(uint64_t new_seed = std::mt19937_64::default_seed) {
        m_rng.seed(new_seed);
    }
    static std::mt19937_64* get_mt() {
        return &m_rng;
    }
    static uint64_t get() {
        return m_rng();
    }
    static int get_uniform_int(const int v_min, const int v_max) {
        std::uniform_int_distribution<int> ud(v_min, v_max);
        return ud(m_rng);
    }
    template<typename TYPE>
    static TYPE get_uniform_real(const TYPE v_min, const TYPE v_max) {
        std::uniform_real_distribution<TYPE> ud(v_min, v_max);
        return ud(m_rng);
    }
};

namespace std {

    template <typename Scalar, int Rows, int Cols>
    struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
        // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
        size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
            size_t seed = 0;
            for (size_t i = 0; i < matrix.size(); ++i) {
                Scalar elem = *(matrix.data() + i);
                seed ^=
                    std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    inline bool operator < (const cv::Vec3b& s1, const cv::Vec3b& s2)
    {
        return s1[0] < s2[0] || (s1[0] == s2[0] && s1[1] < s2[1]) ||(s1[0] == s2[0] && s1[1] == s2[1] && s1[2] < s2[2]);
    }

}  // namespace std

template<typename A>
bool save(const fs::path& v_path, const A& v_obj)
{
    std::ofstream ofs(v_path.string());
    boost::archive::text_oarchive oar(ofs);
    oar << v_obj;
    ofs.close();
    return true;
}

template<typename A>
bool load(const fs::path& v_path, A& v_obj)
{
    std::ifstream ifs(v_path.string());
    if (!ifs.good())
        return false;
    boost::archive::text_iarchive iar(ifs);
    iar >> v_obj;
    ifs.close(); 
    return true;
}

template<typename A>
bool check_and_load(const fs::path& v_path, A& v_obj)
{
    if (fs::exists(v_path))
    {
        if (load(v_path, v_obj))
            return true;
        else
            return false;
    }
    else
        return false;
}

#endif
