#include "inexact_kernel.h"

template<typename A1>
bool is_parallel(const A1& v1, const A1& v2, const double v_eps)
{
	const double angle = cgal_normalized(v1) * cgal_normalized(v2);
	return std::abs(angle) > v_eps;
}

bool is_colinear(const Point_2& p1, const Point_2& p2, const Point_2& p3, const double v_eps)
{
	return is_parallel(p1 - p2, p1 - p3, v_eps);
}

bool is_same_direction(const Vector_2& v1, const Vector_2& v2, const double v_eps)
{
	const double angle = cgal_normalized(v1) * cgal_normalized(v2);
	return angle > v_eps;
}

bool is_oppsite_direction(const Vector_2& v1, const Vector_2& v2, const double v_eps)
{
	const double angle = cgal_normalized(v1) * cgal_normalized(v2);
	return angle < -v_eps;
}

bool is_overlap(const Line_2& v_line1, const Line_2& v_line2)
{
	if (!is_parallel(v_line1.to_vector(), v_line2.to_vector()))
		return false;
	if (std::sqrt(CGAL::squared_distance(v_line1.projection(Point_2(0.,0.)), v_line2)) > NUMERICAL_THRESHOLD)
		return false;
	return true;
}

template <typename A1>
bool is_overlap(const A1& v_point1, const A1& v_point2, const double v_eps)
{
	if (std::sqrt(CGAL::squared_distance(v_point1, v_point2)) > v_eps)
		return false;
	return true;
}

template <typename A1, typename A2, typename A3>
std::pair<bool, A3> intersect(const A1& item1, const A2& item2)
{
	if (!CGAL::do_intersect(item1, item2))
		return { false, A3() };
	if (const auto intersection_result = CGAL::intersection(item1, item2))
	{
		if (const A3* result = boost::get<A3>(&*intersection_result))
			return { true, *result };
		else
			LOG(ERROR) << "!!!!!!!!!!!!!!!!!!!!!!Wrong type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
	}
	else
	{
		LOG(INFO) << "Impossible to reach here";
		//assert(!CGAL::do_intersect(item1, item2));
		return { false, A3() };
	}
}


std::pair<bool, Point_2> intersect_(const Line_2& item1, const Segment_2& item2)
{
	if (!CGAL::do_intersect(item1, item2.supporting_line()))
		return { false, Point_2() };
	Point_2 line_intersection_point;
	if (const auto intersection_result = CGAL::intersection(item1, item2.supporting_line()))
	{
		if (const Point_2* result = boost::get<Point_2>(&*intersection_result))
			line_intersection_point = *result;
		else
			return { false, Point_2() };
	}
	else
		return { false, Point_2() };

	if (std::sqrt((item2.start() - line_intersection_point).squared_length()) < NUMERICAL_THRESHOLD)
		return { true, line_intersection_point };
	else if (std::sqrt((item2.end() - line_intersection_point).squared_length()) < NUMERICAL_THRESHOLD)
		return { true, line_intersection_point };
	else
	{
		if (is_oppsite_direction(line_intersection_point - item2.start(), line_intersection_point - item2.end()))
			return { true, line_intersection_point };
		else
			return { false, Point_2() };
		return intersect<Line_2, Segment_2, Point_2>(item1, item2);
	}
}

std::pair<bool, Point_2> intersect_(const Segment_2& item1, const Line_2& item2)
{
	return intersect_(item2, item1);
}

std::pair<bool, Point_2> intersect_(const Line_2& item1, const Ray_2& item2)
{
	if (!CGAL::do_intersect(item1, item2.supporting_line()))
		return { false, Point_2() };
	Point_2 line_intersection_point;
	if (const auto intersection_result = CGAL::intersection(item1, item2.supporting_line()))
	{
		if (const Point_2* result = boost::get<Point_2>(&*intersection_result))
			line_intersection_point = *result;
		else
			return { false, Point_2() };
	}
	else
		return { false, Point_2() };

	if (std::sqrt((item2.start() - line_intersection_point).squared_length()) < NUMERICAL_THRESHOLD)
		return { true, line_intersection_point };
	else
	{
		if (is_same_direction(line_intersection_point - item2.start(), item2.to_vector()))
			return { true, line_intersection_point };
		else
			return { false, Point_2() };
	}
}

std::pair<bool, std::vector<Point_2>> intersect_(const Line_2& item1, const Polygon_2& item2)
{
	std::vector<Point_2> intersected_point;
	for (int i_edge = 0;i_edge < item2.size();++i_edge)
	{
		const auto& edge = item2.edge(i_edge);
		auto intersection_2d = intersect_(edge, item1);
		if (intersection_2d.first)
		{
			intersected_point.push_back(intersection_2d.second);
		}
	}
	return { !intersected_point.empty(), intersected_point };
}

template bool is_parallel(const Vector_2& v1, const Vector_2& v2, const double v_eps);
template bool is_parallel(const Vector_3& v1, const Vector_3& v2, const double v_eps);

template bool is_overlap(const Point_2& v_point1, const Point_2& v_point2, const double v_eps);
template bool is_overlap(const Point_3& v_point1, const Point_3& v_point2, const double v_eps);

template std::pair<bool, Point_2> intersect<Line_2, Line_2, Point_2>(const Line_2& item1, const Line_2& item2);
template std::pair<bool, Line_3> intersect<Plane_3, Plane_3, Line_3>(const Plane_3& item1, const Plane_3& item2);
