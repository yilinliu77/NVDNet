#pragma once

/*
 * Copyright (C) 2016-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef GEOM_ICP_HEADER
#define GEOM_ICP_HEADER

#include <limits>
#include <random>
#include <future>
#include <vector>
#include <cassert>
#include <algorithm>
#include <unordered_set>

#include "cgal_tools.h"
#include "intersection_tools.h"

#include "pointmatcher/PointMatcher.h"

constexpr float eps = std::numeric_limits<float>::epsilon();

typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> Correspondence;
typedef std::vector<Correspondence> Correspondences;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

std::vector<uint> choose_random(std::size_t n, std::size_t from, std::size_t to) {
    static std::mt19937 gen;
    std::uniform_int_distribution<uint> dis(from, to);
    std::unordered_set<uint> samples;
    while(samples.size() < std::min(n, to - from)) {
        samples.insert(dis(gen));
    }
    return std::vector<uint>(samples.begin(), samples.end());
}

std::pair<Eigen::Matrix3d, float> determine_rotation_and_scale(const Eigen::Matrix3d& cov, double sigma2) {
    Eigen::Matrix3d U, S, V;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U=svd.matrixU();
    S=svd.singularValues().asDiagonal();
    V=svd.matrixV();
    Eigen::Matrix3d VUt = V * U.transpose();
    
    double det = VUt.determinant();
    if (det > 0.0) {
        double scale = S.trace() / sigma2;
        return {VUt, 1.f};
    } else {
        Eigen::Matrix3d F;
        F.fill(0.f);
        F(0, 0) = 1.0; F(1, 1) = 1.0; F(2, 2) = -1.0;
        double scale = (S * F).trace() / sigma2;
        return {V * F * U.transpose(), scale};
    }
}

std::pair<Eigen::Matrix3d, float>
determine_rotation_and_scale(std::vector<Correspondence> const & ccorrespondences)
{
    Eigen::Matrix3d cov;
    cov.fill(0.f);
    double sigma2 = 0.0f;

    for (std::size_t i = 0; i < ccorrespondences.size(); ++i) {
        Eigen::Vector3d f, c;
        std::tie(f, c) = ccorrespondences[i];
        cov += Eigen::Matrix<double,3,1>(f.x(),f.y(),f.z()) *
           Eigen::Matrix<double,1,3>(c.x(),c.y(),c.z());
        
        sigma2 += f.squaredNorm();
    }
    cov /= ccorrespondences.size();
    sigma2 /= ccorrespondences.size();

    return determine_rotation_and_scale(cov, sigma2);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> center(std::vector<Correspondence> * correspondences) {
    Eigen::Vector3d c0(0.0f,0.0f,0.0f), c1(0.0f,0.0f,0.0f);
    for (std::size_t i = 0; i < correspondences->size(); ++i) {
        Eigen::Vector3d f, c;
        std::tie(f, c) = correspondences->at(i);
        c0 += f;
        c1 += c;
    }
    c0 /= correspondences->size();
    c1 /= correspondences->size();

    for (std::size_t i = 0; i < correspondences->size(); ++i) {
        Eigen::Vector3d f, c;
        std::tie(f, c) = correspondences->at(i);
        correspondences->at(i).first = f - c0;
        correspondences->at(i).second = c - c1;
    }

    return std::make_pair(c0, c1);
}

Eigen::Matrix4d estimate_transform(std::vector<Correspondence> correspondences)
{
    if (correspondences.empty()) throw std::runtime_error("No correspondences");

    Eigen::Vector3d c0, c1;
    std::tie(c0, c1) = center(&correspondences);

    Eigen::Matrix3d R;
    float s;
    std::tie(R, s) = determine_rotation_and_scale(correspondences);
    Eigen::Vector3d t = c1 + (-R * s * c0);

    Eigen::Matrix4d result;
    result.fill(0.f);
    result.block(0,0,3,3) = R * s;
    result.block(0,3,3,1) = t;
    result(3,3)=1.f;
    return result;
}

Eigen::Matrix4d estimate_transform1(std::vector<Correspondence>& correspondences)
{
	Eigen::Vector3d p1(0.f,0.f,0.f), p2(0.f,0.f,0.f);
    int N = correspondences.size();
    for (int i=0; i<N; i++)
    {
        p1 += correspondences[i].first;
        p2 += correspondences[i].second;
    }
    p1 /= N;
    p2 /= N;

    // subtract COM
    std::vector<Eigen::Vector3d> q1(N), q2(N);
    for (int i=0; i<N; i++)
    {
        q1[i] = correspondences[i].first - p1;
        q2[i] = correspondences[i].second - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i=0; i<N; i++)
        W += Eigen::Vector3d(q1[i].x(), q1[i].y(), q1[i].z()) * Eigen::Vector3d(q2[i].x(),q2[i].y(), q2[i].z()).transpose();
	//std::cout << "W=" << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    //cout << "U=" << U << endl;
    //cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x(), p1.y(), p1.z()) - R_ * Eigen::Vector3d(p2.x(), p2.y(), p2.z());

    // convert to cv::Mat
    Eigen::Matrix4d result;
    result.fill(0.f);
    result(3,3)=1.f;
   result.block(0,0,3,3) <<
            R_(0, 0), R_(0, 1), R_(0,2),
            R_(1, 0), R_(1, 1), R_(1,2),
            R_(2, 0), R_(2, 1), R_(2,2);
    result.block(0,3,3,1) << t_(0, 0), t_(1, 0), t_(2, 0);
    return result.inverse();
}

template <typename TreeType1,typename TreeType2>
Eigen::Matrix4d estimate_transform(const Point_set& v_source_points, const Point_set& v_target_points,
    uint num_iters, double& avg_dist_ptr, const TreeType1* v_source_tree=nullptr, const TreeType2* v_target_tree=nullptr, float square_distance_threshold = 1e-6f)
{
    assert(num_iters < std::numeric_limits<uint>::max() - 1);

    KDTree *kdtree_source, *kdtree_target;

    if(v_source_tree==nullptr)
    {
        kdtree_source=new KDTree(v_source_points.points().begin(), v_source_points.points().end());
        kdtree_source->build();
    }
    if(v_target_tree==nullptr)
    {
        kdtree_target=new KDTree(v_target_points.points().begin(), v_target_points.points().end());
        kdtree_target->build();
    }


    Eigen::Matrix4d best_T;
    Eigen::Matrix4d T;
    T.fill(0);
    T(0, 0) = T(1, 1) = T(2, 2) = T(3, 3) = 1.0;
    double prev_avg_sqdist, avg_sqdist = std::numeric_limits<double>::max();

    std::size_t num_verts = v_source_points.size() + v_target_points.size();

    double threshold = 0.25f;

    Correspondences corrs(num_verts);
    Correspondences valid_corrs;
    std::vector<uint8_t> valid(num_verts);
    double best_distance = 99999;
    const int MAX_TOLERENCE = 3;
    int tolerence = MAX_TOLERENCE;
    for (uint i_iter = 0; i_iter < num_iters + 1; ++i_iter) {
        Eigen::Matrix4d Ti = T.inverse();

        prev_avg_sqdist = avg_sqdist;
        avg_sqdist = 0.0;
        std::fill(valid.begin(), valid.end(), 0);
        valid_corrs.clear();
        valid_corrs.reserve(num_verts);

        /* Find correspondences */
        #pragma omp parallel for reduction(+:avg_sqdist)
        for (int j = 0; j < v_source_points.size(); ++j) {
            Eigen::Vector4d vertex4 = T * Eigen::Vector4d(v_source_points.point(j).x(),v_source_points.point(j).y(),v_source_points.point(j).z(),1);
            Point_3 vertex(vertex4.x()/vertex4.w(),vertex4.y()/vertex4.w(),vertex4.z()/vertex4.w());
            Point_3 cp;
        	if(v_target_tree)
				cp = v_target_tree->closest_point(vertex);
            else
            {
	            Neighbor_search search(*kdtree_target, vertex, 1);
                cp = search.begin()->first;
            }
            double dist = std::sqrt((vertex - cp).squared_length());
            if (dist < threshold) {
                avg_sqdist += dist * dist;
                corrs[j] = std::make_pair(Eigen::Vector3d(vertex.x(),vertex.y(),vertex.z()), Eigen::Vector3d(cp.x(),cp.y(),cp.z()));
                valid[j] = 255;
            }
        }
        #pragma omp parallel for reduction(+:avg_sqdist)
        for (int j = 0; j < v_target_points.size(); ++j) {
            Eigen::Vector4d vertex4 = Ti * Eigen::Vector4d(
                v_target_points.point(j).x(),
                v_target_points.point(j).y(),
                v_target_points.point(j).z(),
                1.f);
            Point_3 vertex(vertex4.x()/vertex4.w(),vertex4.y()/vertex4.w(),vertex4.z()/vertex4.w());
            Point_3 cp;
            if(v_source_tree)
				cp = v_source_tree->closest_point(vertex);
            else
            {
	            Neighbor_search search(*kdtree_source, vertex, 1);
                cp = search.begin()->first;
            }
            double dist = std::sqrt((vertex - cp).squared_length());
            if (dist < threshold) {
                avg_sqdist += dist * dist;
                corrs[v_source_points.size() + j] = std::make_pair(Eigen::Vector3d(vertex.x(),vertex.y(),vertex.z()), Eigen::Vector3d(cp.x(),cp.y(),cp.z()));
                valid[v_source_points.size() + j] = 255;
            }
        }

        for (std::size_t j = 0; j < num_verts; ++j) {
            if (valid[j] == 255) {
                valid_corrs.push_back(corrs[j]);
            }
        }
        Eigen::Matrix<double, 3, Eigen::Dynamic> source_eigen(3,valid_corrs.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> target_eigen(3,valid_corrs.size());
        for (std::size_t j = 0; j < valid_corrs.size(); ++j) 
        {
            source_eigen(0,j) = valid_corrs[j].first.x();
            source_eigen(1,j) = valid_corrs[j].first.y();
            source_eigen(2,j) = valid_corrs[j].first.z();
            target_eigen(0,j) = valid_corrs[j].second.x();
            target_eigen(1,j) = valid_corrs[j].second.y();
            target_eigen(2,j) = valid_corrs[j].second.z();
        }

        avg_sqdist /= static_cast<double>(valid_corrs.size());

        std::cout << avg_sqdist << ' ' << static_cast<double>(valid_corrs.size()) / num_verts << std::endl;
        double improvement = prev_avg_sqdist - avg_sqdist;
        if (improvement < 1e-5f) break;

   //     if(best_distance<avg_sqdist) // Unsucessful transformation
   //     {
   //         tolerence--;
   //     }
   //     else if(best_distance < avg_sqdist + square_distance_threshold) // Sucessful transformation, but the improvement is too little
   //     {
	  //      tolerence--;
			//best_distance = std::min(avg_sqdist,best_distance);
	  //      best_T = T;
   //     }
   //     else if(best_distance>avg_sqdist)  // Sucessful transformation, reset the tolerence
   //     {
			//best_distance = std::min(avg_sqdist,best_distance);
   //         tolerence = MAX_TOLERENCE;
	  //      best_T = T;
   //     }
   //     if (tolerence < 0 || i_iter == num_iters) break;

        T = Eigen::umeyama(source_eigen,target_eigen,true) * T;

        //T = estimate_transform(valid_corrs) * T;
    }

    avg_dist_ptr = std::sqrt(avg_sqdist);

    return T;
}

template <typename TreeType1, typename TreeType2>
void prepare_correspondance(const Point_set& v_source_points, const Point_set& v_target_points, 
    const TreeType1* v_source_tree, const TreeType2* v_target_tree, DP& v_recon, DP& v_ref)
{
    DP::Labels featLabels;
    featLabels.push_back(DP::Label("x", 1));
    featLabels.push_back(DP::Label("y", 1));
    featLabels.push_back(DP::Label("z", 1));
    featLabels.push_back(DP::Label("pad", 1));
    DP::Labels descLabels;
    descLabels.push_back(DP::Label("normal", 3));

    PM::Matrix recon_points;
    PM::Matrix ref_points;

    const bool JUST_USE_SAMPLE_POINT = false;
    if (JUST_USE_SAMPLE_POINT)
    {
        recon_points = PM::Matrix(4, v_source_points.size());
        ref_points = PM::Matrix(4, v_target_points.size());
    }
    else
    {
        recon_points = PM::Matrix(4, v_source_points.size() + v_target_points.size());
        ref_points = PM::Matrix(4, v_source_points.size() + v_target_points.size());
    }

#pragma omp parallel for
    for (int j = 0; j < v_source_points.size(); ++j) {
        if(JUST_USE_SAMPLE_POINT)
        {
            recon_points(0, j) = v_source_points.point(j).x();
            recon_points(1, j) = v_source_points.point(j).y();
            recon_points(2, j) = v_source_points.point(j).z();
            recon_points(3, j) = 1;
        }
        else
        {
            Point_3 cp;
            cp = v_target_tree->closest_point(v_source_points.point(j));
            recon_points(0, j) = v_source_points.point(j).x();
            recon_points(1, j) = v_source_points.point(j).y();
            recon_points(2, j) = v_source_points.point(j).z();
            recon_points(3, j) = 1;

            ref_points(0, j) = cp.x();
            ref_points(1, j) = cp.y();
            ref_points(2, j) = cp.z();
            ref_points(3, j) = 1;
        }


    }
#pragma omp parallel for
    for (int j = 0; j < v_target_points.size(); ++j) {
        if (JUST_USE_SAMPLE_POINT)
        {
            ref_points(0, j) = v_target_points.point(j).x();
            ref_points(1, j) = v_target_points.point(j).y();
            ref_points(2, j) = v_target_points.point(j).z();
            ref_points(3, j) = 1;
        }
        else
        {
            Point_3 cp;
            cp = v_source_tree->closest_point(v_target_points.point(j));

            ref_points(0, v_source_points.size() + j) = v_target_points.point(j).x();
            ref_points(1, v_source_points.size() + j) = v_target_points.point(j).y();
            ref_points(2, v_source_points.size() + j) = v_target_points.point(j).z();
            ref_points(3, v_source_points.size() + j) = 1;

            recon_points(0, v_source_points.size() + j) = cp.x();
            recon_points(1, v_source_points.size() + j) = cp.y();
            recon_points(2, v_source_points.size() + j) = cp.z();
            recon_points(3, v_source_points.size() + j) = 1;
        }
        
    }

    v_recon = DP(recon_points,featLabels);
    v_ref = DP(ref_points, featLabels);
    return;
}

template <typename TreeType1, typename TreeType2>
Eigen::Matrix4d estimate_transform_plus(const Point_set& v_source_points, const Point_set& v_target_points,
    uint num_iters, double& avg_dist_ptr, const TreeType1* v_source_tree = nullptr, const TreeType2* v_target_tree = nullptr, float square_distance_threshold = 0.25f)
{
    assert(num_iters < std::numeric_limits<uint>::max() - 1);

    Eigen::Matrix4d T;
    T.fill(0);
    T(0, 0) = T(1, 1) = T(2, 2) = T(3, 3) = 1.0;

    // Setup libpointmatcher
    DP recon, ref;
    PM::ICP icp;
    {
        prepare_correspondance(v_source_points, v_target_points, v_source_tree, v_target_tree, recon, ref);
        icp.setDefault();

        PointMatcherSupport::Parametrizable::Parameters params;
        std::string name;

        // Checker
        name = "DifferentialTransformationChecker";
        params["minDiffRotErr"] = "0.00001";
        params["minDiffTransErr"] = "0.00001";
        params["smoothLength"] = "4";
        std::shared_ptr<PM::TransformationChecker> diff =
            PM::get().TransformationCheckerRegistrar.create(name, params);
        params.clear();

        // Prepare outlier filters
        name = "MaxDistOutlierFilter";
        params["maxDist"] = std::to_string(square_distance_threshold);
        std::shared_ptr<PM::OutlierFilter> trim =
            PM::get().OutlierFilterRegistrar.create(name, params);
        params.clear();

        icp.transformationCheckers.clear();
        icp.transformationCheckers.push_back(diff);

        icp.outlierFilters.clear();
        icp.outlierFilters.push_back(trim);

        PM::TransformationParameters icp_result = icp(recon, ref);

        float matchRatio = icp.errorMinimizer->getWeightedPointUsedRatio();
        float overlap = icp.errorMinimizer->getOverlap();

        DP data_out(recon);
        icp.transformations.apply(data_out, icp_result);
        icp.matcher->init(ref);
        PM::Matches matches = icp.matcher->findClosests(data_out);
        PM::OutlierWeights outlierWeights = icp.outlierFilters.compute(data_out, ref, matches);
        float error = 0;
        for (int i = 0;i < recon.getNbPoints();++i)
            if (outlierWeights(0, i) == 1.f)
                error += matches.dists(0, i);
        error /= recon.getNbPoints();
        LOG(INFO) << boost::format("%f points useds; Overlap is %f; Error is %f") % matchRatio % overlap % error;
        T = icp_result.cast<double>();
        avg_dist_ptr = error;
    }
    return T;
}


#endif /* GEOM_ICP_HEADER */