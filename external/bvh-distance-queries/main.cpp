#include<iostream>
#include<tuple>

#include "src/bvh.h"


int main()
{
    std::vector<double> triangles { 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1 };
    std::vector<double> points { -1, -1, -1, 3, 3, 3 };

    // auto results = bvh_distance_queries(triangles, points);

    return 1;
}