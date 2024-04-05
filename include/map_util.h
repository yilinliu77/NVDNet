#pragma once
/*
 * Convert coordinates in airsim, unreal, mesh, image
 */
#include "common_util.h"

#include<Eigen/Dense>

struct Pos_Pack
{
	Eigen::Vector3f pos_mesh;
	Eigen::Vector3f pos_airsim;
	Eigen::Vector3f direction;
	Eigen::Isometry3f camera_matrix;

	float yaw, pitch;
};

class MapConverter {
public:
    MapConverter();

    Eigen::Vector3f mDroneStart;

    float mImageStartX;
    float mImageStartY;

    bool initDroneDone = false;
    bool initImageDone = false;

    void initDroneStart(const Eigen::Vector3f& vPos);

    Eigen::Vector3f convertUnrealToAirsim(const Eigen::Vector3f& vWorldPos) const;

    Eigen::Vector3f convertUnrealToMesh(const Eigen::Vector3f& vWorldPos) const;

    Eigen::Vector3f convertMeshToUnreal(const Eigen::Vector3f& vMeshPos) const;

    Eigen::Vector3f convertAirsimToMesh(const Eigen::Vector3f& vAirsimPos) const;
	
    Eigen::Matrix3f convert_yaw_pitch_to_matrix_mesh(const float yaw, const float pitch);

    static Eigen::Isometry3f get_camera_matrix(const float yaw, const float pitch, const Eigen::Vector3f& v_pos);

    static Eigen::Vector3f convert_yaw_pitch_to_direction_vector(const float yaw, const float pitch);

    static Pos_Pack get_pos_pack_from_direction_vector(const Eigen::Vector3f& v_pos_mesh, const Eigen::Vector3f& v_direction);
    Pos_Pack get_pos_pack_from_unreal(const Eigen::Vector3f& v_pos_unreal, float yaw, float pitch);
    Pos_Pack get_pos_pack_from_mesh(const Eigen::Vector3f& v_pos_mesh, float v_mesh_yaw, float v_mesh_pitch);

};