#pragma once
#ifndef ELLIPSE_FIT_H
#define ELLIPSE_FIT_H


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;


class Ellipse_fit
{
public:
	void set(vector<vector<double>> input);
	void fit(double& result_center_x, double& result_center_y, double& result_phi, double& result_width,
	         double& result_hight);

private:
	vector<vector<double>> input_matrix;
	void take_input(vector<vector<double>> input);
};
#endif
