#include "ellipse_fit.h"


void Ellipse_fit::set(vector<vector<double>> input)
{
	take_input(input);
}


void Ellipse_fit::take_input(vector<vector<double>> input)
{
	input_matrix = input;
}


void Ellipse_fit::fit(double& result_center_x, double& result_center_y, double& result_phi, double& result_width,
                      double& result_hight)
{
	size_t size_mat = input_matrix.size();

	Eigen::VectorXd X_val(size_mat);
	Eigen::VectorXd Y_val(size_mat);

	for (size_t k = 0; k < size_mat; k++)
	{
		X_val(k) = input_matrix[k][0];
		Y_val(k) = input_matrix[k][1];
	}


	Eigen::VectorXd D1_col0(size_mat);
	Eigen::VectorXd D1_col1(size_mat);
	Eigen::VectorXd D1_col2(size_mat);

	D1_col0 = X_val.array().pow(2);
	D1_col1 = X_val.array() * Y_val.array();
	D1_col2 = Y_val.array().pow(2);


	Eigen::MatrixXd D1(size_mat, 3);

	D1.col(0) = D1_col0;
	D1.col(1) = D1_col1;
	D1.col(2) = D1_col2;

	Eigen::MatrixXd D2(size_mat, 3);

	D2.col(0) = X_val;
	D2.col(1) = Y_val;
	D2.col(2) = Eigen::VectorXd::Ones(size_mat);


	Eigen::MatrixXd S1(3, 3);
	Eigen::MatrixXd S2(3, 3);
	Eigen::MatrixXd S3(3, 3);


	S1 = D1.transpose() * D1;
	S2 = D1.transpose() * D2;
	S3 = D2.transpose() * D2;


	Eigen::MatrixXd C1(3, 3);
	C1 << 0, 0, 2, 0, -1, 0, 2, 0, 0;

	Eigen::MatrixXd M;

	M = C1.inverse() * (S1 - S2 * S3.inverse() * S2.transpose());

	Eigen::EigenSolver<MatrixXd> s(M);


	MatrixXd eigenvector = s.eigenvectors().real();


	Eigen::VectorXd eig_row0 = eigenvector.row(0);
	Eigen::VectorXd eig_row1 = eigenvector.row(1);
	Eigen::VectorXd eig_row2 = eigenvector.row(2);

	Eigen::VectorXd cond = 4 * (eig_row0.array() * eig_row2.array()) - eig_row1.array().pow(2);


	Eigen::VectorXd min_pos_eig;

	for (int i = 0; i < 3; i++)
	{
		if (cond(i) > 0)
		{
			min_pos_eig = eigenvector.col(i);
			break;
		}
	}
	Eigen::VectorXd coeffs(6);
	Eigen::VectorXd cont_matrix = -1 * S3.inverse() * S2.transpose() * min_pos_eig;
	coeffs << min_pos_eig, cont_matrix;


	double a = coeffs(0);
	double b = coeffs(1) / 2;
	double c = coeffs(2);
	double d = coeffs(3) / 2;
	double f = coeffs(4) / 2;
	double g = coeffs(5);

	double center_x = (c * d - b * f) / (pow(b, 2) - a * c);
	double center_y = (a * f - b * d) / (pow(b, 2) - a * c);

	double numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g);
	double denominator1 = (b * b - a * c) * ((c - a) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a));
	double denominator2 = (b * b - a * c) * ((a - c) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a));
	double width = sqrt(numerator / denominator1);
	double height = sqrt(numerator / denominator2);
	double phi = 0.5 * atan((2 * b) / (a - c));


	result_center_x = center_x;
	result_center_y = center_y;
	result_width = width;
	result_hight = height;
	result_phi = phi;
}
