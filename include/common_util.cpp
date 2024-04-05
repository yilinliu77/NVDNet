#include "common_util.h"

#include <argparse/argparse.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <string>
bool interaction_on = true;

#ifdef _WIN32
#include <time.h>
#else
#include <time.h>
#endif

std::mt19937_64 Global_helper::m_rng;

Json::Value setup_args(int argc, char** argv, const std::string& v_name)
{
	Json::Value args;
	argparse::ArgumentParser program(v_name);

	if (argc < 1)
	{
		std::cerr << "Need to input a json config file" << std::endl;
		return -1;
	}
	else if (!fs::exists(argv[1]))
	{
		std::cerr << "Config file " << argv[1] << "not exists" << std::endl;
		return -1;
	}
	else
	{
		const std::string config_file = argv[1];

		std::ifstream in(config_file);
		if (!in.is_open())
		{
			std::cerr << "Error opening file" << config_file << std::endl;
			return 0;
		}
		Json::Reader json_reader;
		if (!json_reader.parse(in, args))
		{
			std::cerr << "Error parse config file" << config_file << std::endl;
			return 0;
		}
		in.close();
	}
	program.add_argument("config_file").required();

	for (const auto& arg_name : args.getMemberNames())
		program.add_argument("--" + arg_name);

	program.parse_args(argc, argv);
	for (const auto& arg_name : args.getMemberNames())
	{
		try
		{
			if (!program.is_used("--" + arg_name))
				continue;

			const auto& value = program.get<std::string>("--" + arg_name);

			if (args[arg_name].type() == Json::booleanValue)
			{
				std::cout << "Overide " << arg_name << " from " << args[arg_name].asString();
				if (value == "false")
					args[arg_name] = false;
				else if (value == "true")
					args[arg_name] = true;
				else if (value == "0")
					args[arg_name] = false;
				else if (value == "1")
					args[arg_name] = true;
				else
					throw;
				std::cout << " to " << args[arg_name].asString() << " from command line" << std::endl;
			}
			else if (args[arg_name].type() == Json::realValue)
			{
				std::cout << "Overide " << arg_name << " from " << args[arg_name].asString();
				args[arg_name] = std::atof(value.c_str());
				std::cout << " to " << args[arg_name].asString() << " from command line" << std::endl;
			}
			else if (args[arg_name].type() == Json::intValue)
			{
				std::cout << "Overide " << arg_name << " from " << args[arg_name].asString();
				args[arg_name] = (int)std::atof(value.c_str());
				std::cout << " to " << args[arg_name].asString() << " from command line" << std::endl;
			}
			else if (args[arg_name].type() == Json::arrayValue)
			{
				std::cout << "Overide " << arg_name << " from ";
				for (int i = 0; i < args[arg_name].size(); ++i)
					std::cout << args[arg_name][i].asString() << ", ";

				Json::Value values;
				std::vector<std::string> tokens;
				boost::split(tokens, value, boost::is_any_of(","));
				for (size_t i = 0; i < args[arg_name].size(); ++i)
					values.append(std::atof(tokens[i].c_str()));
				args[arg_name] = values;

				std::cout << " to ";
				for (int i = 0; i < args[arg_name].size(); ++i)
					std::cout << args[arg_name][i].asString() << ", ";
				std::cout << std::endl;
			}
			else
			{
				std::cout << "Overide " << arg_name << " from " << args[arg_name].asString();
				args[arg_name] = value.c_str();
				std::cout << " to " << args[arg_name].asString() << " from command line" << std::endl;
			}
		}
		catch (...)
		{
		}
	}

	return args;
}

void checkFolder(const fs::path& folder)
{
	if (fs::is_directory(folder))
	{
		fs::remove_all(folder);
	}
	fs::create_directories(folder);
}

void safeCheckFolder(const fs::path& folder)
{
	if (!fs::is_directory(folder))
	{
		fs::create_directories(folder);
	}
}

std::chrono::steady_clock::time_point recordTime()
{
	std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
	return now;
}

double profileTime(std::chrono::steady_clock::time_point& now, std::string vTip, bool v_profile)
{
	auto t2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);
	if (v_profile)
		std::cout << vTip << ": " << time_span.count() << std::endl;
	now = std::chrono::steady_clock::now();
	return time_span.count();
}

float getTime(std::chrono::steady_clock::time_point& now)
{
	auto t2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);
	return time_span.count();
}

void debug_img(const cv::Mat& v_img)
{
	cv::imshow("Debug", v_img);
	cv::waitKey(0);
}

void debug_img(std::vector<cv::Mat>& vImgs)
{
	cv::namedWindow("Debug", cv::WINDOW_NORMAL);
	cv::resizeWindow("Debug", 800 * static_cast<int>(vImgs.size()), 800);
	cv::Mat totalImg;

	cv::hconcat(vImgs, totalImg);

	cv::imshow("Debug", totalImg);
	cv::waitKey(0);
	cv::destroyWindow("Debug");
}

void wait_interaction()
{
	if (interaction_on)
	{
		std::vector<cv::Mat> img{cv::Mat(1, 1,CV_8UC1)};
		debug_img(img);
	}
}

void override_sleep(float seconds)
{
#ifdef _WIN32
	_sleep(static_cast<unsigned long>(seconds * 1000));
#else
	sleep(seconds);
#endif
}

std::vector<cv::Vec3b> get_color_table_bgr()
{
	std::vector<cv::Vec3b> color_table;
	//color_table.emplace_back(197, 255, 255);
	//color_table.emplace_back(226, 226, 255);
	//color_table.emplace_back(255, 226, 197);
	//color_table.emplace_back(197, 255, 226);
	//color_table.emplace_back(2, 0, 160);
	//color_table.emplace_back(0, 12, 79);
	//color_table.emplace_back(105, 72, 129);
	//color_table.emplace_back(153, 0, 102);
	//color_table.emplace_back(153, 150, 102);

	//color_table.emplace_back(0, 0, 0);
	//color_table.emplace_back(0, 255, 0);
	//color_table.emplace_back(0, 0, 255);
	//color_table.emplace_back(2, 0, 104);
	//color_table.emplace_back(51, 0, 255);
	//color_table.emplace_back(51, 102, 255);
	//color_table.emplace_back(47, 84, 227);
	//color_table.emplace_back(25, 0, 203);
	//color_table.emplace_back(38, 137, 243);
	//color_table.emplace_back(8, 69, 231);
	//color_table.emplace_back(41, 160, 252);
	//color_table.emplace_back(0, 102, 255);
	//color_table.emplace_back(0, 206, 255);
	//color_table.emplace_back(24, 104, 235);

	color_table.emplace_back(153, 255, 204);
	color_table.emplace_back(255, 204, 153);
	color_table.emplace_back(153, 255, 255);
	color_table.emplace_back(253, 196, 225);
	color_table.emplace_back(0, 182, 246);

	return color_table;
}

std::vector<cv::Vec3b> get_color_table_bgr2()
{
	std::vector<cv::Vec3b> color_table;
	color_table.emplace_back(197, 255, 255);
	color_table.emplace_back(226, 226, 255);
	color_table.emplace_back(255, 226, 197);
	color_table.emplace_back(197, 255, 226);
	color_table.emplace_back(2, 0, 160);
	color_table.emplace_back(0, 12, 79);
	color_table.emplace_back(105, 72, 129);
	color_table.emplace_back(153, 0, 102);
	color_table.emplace_back(153, 150, 102);

	color_table.emplace_back(153, 255, 204);
	color_table.emplace_back(255, 204, 153);
	color_table.emplace_back(153, 255, 255);
	color_table.emplace_back(253, 196, 225);
	color_table.emplace_back(0, 182, 246);

	return color_table;
}

std::vector<cv::Vec3b> get_color_table_bgr3()
{
	std::vector<cv::Vec3b> color_table;
	color_table.emplace_back(153, 15, 15);
	color_table.emplace_back(178, 44, 44);
	color_table.emplace_back(204, 81, 81);
	color_table.emplace_back(229, 126, 126);
	color_table.emplace_back(255, 178, 178);
	color_table.emplace_back(153, 84, 15);
	color_table.emplace_back(178, 111, 44);
	color_table.emplace_back(204, 142, 81);
	color_table.emplace_back(229, 177, 126);
	color_table.emplace_back(255, 216, 178);
	color_table.emplace_back(107, 153, 15);
	color_table.emplace_back(133, 178, 44);
	color_table.emplace_back(163, 204, 81);
	color_table.emplace_back(195, 229, 126);
	color_table.emplace_back(229, 255, 178);
	color_table.emplace_back(15, 107, 153);
	color_table.emplace_back(44, 133, 178);
	color_table.emplace_back(81, 163, 204);
	color_table.emplace_back(126, 195, 229);
	color_table.emplace_back(178, 229, 255);
	color_table.emplace_back(38, 15, 153);
	color_table.emplace_back(66, 44, 178);
	color_table.emplace_back(101, 81, 204);
	color_table.emplace_back(143, 126, 229);
	color_table.emplace_back(191, 178, 255);
	return color_table;
}

std::vector<cv::Vec3b> get_color_table_bgr4()
{
	std::vector<cv::Vec3b> color_table;
	color_table.emplace_back(96,189,246);
	color_table.emplace_back(226, 237, 247);
	color_table.emplace_back(195, 202, 245);
	color_table.emplace_back(157, 165, 132);
	color_table.emplace_back(130, 132, 242);
	return color_table;
}
