import math
import os.path
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from common_utils import export_point_cloud, check_dir

try:
    sys.path.append("thirdparty")
    import cuda_distance
    import open3d as o3d
    import faiss
except:
    print("Cannot import cuda_distance, ignore this if you don't use 'ABC_dataset_test_mesh'")


def de_normalize_angles(v_angles):
    if isinstance(v_angles,torch.Tensor):
        angles = (v_angles / 65535 * torch.pi * 2)
        dx = torch.cos(angles[..., 0]) * torch.sin(angles[..., 1])
        dy = torch.sin(angles[..., 0]) * torch.sin(angles[..., 1])
        dz = torch.cos(angles[..., 1])
        gradients = torch.stack([dx, dy, dz], dim=-1)
    else:
        angles = (v_angles / 65535 * np.pi * 2)
        dx = np.cos(angles[..., 0]) * np.sin(angles[..., 1])
        dy = np.sin(angles[..., 0]) * np.sin(angles[..., 1])
        dz = np.cos(angles[..., 1])
        gradients = np.stack([dx, dy, dz], axis=-1)
    return gradients


def de_normalize_udf(v_udf):
    return v_udf / 65535 * 2

def normalize_points(v_points):
    min_xyz = v_points.min(axis=0)
    max_xyz = v_points.max(axis=0)
    diag = np.linalg.norm(max_xyz - min_xyz)
    center_xyz = (min_xyz + max_xyz) / 2
    v_points = (v_points - center_xyz[None, :]) / diag * 2
    return v_points


def generate_coords(v_resolution):
    coords = np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution), indexing="ij")
    coords = np.stack(coords, axis=3) / (v_resolution - 1)
    coords = (coords * 2 - 1).astype(np.float32)
    return coords


def angle2vector(v_angles):
    angles = (v_angles / 65535 * np.pi * 2)
    dx = np.cos(angles[..., 0]) * np.sin(angles[..., 1])
    dy = np.sin(angles[..., 0]) * np.sin(angles[..., 1])
    dz = np.cos(angles[..., 1])
    gradients = np.stack([dx, dy, dz], axis=-1)
    return gradients


class ABC_patch_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch_pc, self).__init__()
        self.data_root = v_data_root
        self.mode = v_training_mode
        self.conf = v_conf
        self.mini_batch_size = v_conf["mini_batch_size"]
        # self.pre_check_data()
        with h5py.File(self.data_root, "r") as f:
            assert f["points"].shape[1] % self.mini_batch_size == 0
            self.num_mini_batch = f["points"].shape[1] // self.mini_batch_size
            self.num_items = f["points"].shape[0]
            self.names = np.asarray(
                ["{:08d}".format(f["names"][i]) for i in range(f["names"].shape[0])])
        self.valid_flag = np.load(Path(self.data_root).parent/"precheck_flags.npy") # Items that have no invalid points
        self.actual_num_items = np.sum(self.valid_flag)
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        self.is_bool_flag = self.conf["is_bool_flag"]

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(self.num_mini_batch), indexing="ij"), axis=2)
            self.valid_flag = self.valid_flag[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_mini_batch), indexing="ij"), axis=2)
            self.valid_flag = self.valid_flag[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_mini_batch), indexing="ij"), axis=2)
        else:
            raise ""
        self.index = self.index[self.valid_flag].reshape((-1, 2))


    def pre_check_data(self):
        precheck_flags = []
        with h5py.File(self.data_root, "r") as f:
            num_items = f["point_flags"].shape[0]
            for i in tqdm(range(num_items)):
                flags = (f["point_flags"][i] == 0).all(axis=1)
                precheck_flags.append(flags)
        precheck_flags = np.stack(precheck_flags)
        precheck_flags = np.logical_not(precheck_flags.any(axis=1))
        np.save("precheck_flags", precheck_flags)
        return

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()

        with h5py.File(self.data_root, "r") as f:
            points = f["points"][
                     v_id_item, v_id_patch * self.mini_batch_size:(v_id_patch + 1) * self.mini_batch_size]
            point_flags = f["point_flags"][
                          v_id_item, v_id_patch * self.mini_batch_size:(v_id_patch + 1) * self.mini_batch_size]
            voronoi_flags = f["voronoi_flags"][
                            v_id_item, v_id_patch * self.mini_batch_size:(v_id_patch + 1) * self.mini_batch_size]

            shifts = np.arange(8)[None,:].repeat(self.mini_batch_size, axis=0)
            point_flags = ((point_flags[:, :, None] & (1 << shifts)[:, None, :]) > 0).reshape(
                self.mini_batch_size, -1)
            voronoi_flags = ((voronoi_flags[:, :, None] & (1 << shifts)[:, None, :]) > 0).reshape(
                self.mini_batch_size, 1, 32, 32, 32).astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()

        return points, point_flags, voronoi_flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        points, point_flags, voronoi_flags = self.get_patch(id_object, id_patch)
        return np.concatenate([points, point_flags[:,:,None].astype(np.float32)],axis=-1), voronoi_flags, \
            self.names[id_object], np.arange(point_flags.shape[0], dtype=np.int64) + id_patch * point_flags.shape[0]

    @staticmethod
    def collate_fn(v_batches):
        feat_data, flag_data, names, id_patch = [], [], [], []
        for item in v_batches:
            feat_data.append(item[0])
            flag_data.append(item[1])
            names.append(item[2])
            id_patch.append(item[3])
        feat_data = np.stack(feat_data, axis=0)
        flag_data = np.stack(flag_data, axis=0)
        id_patch = np.stack(id_patch, axis=0)
        names = np.asarray(names)

        return (
            (torch.from_numpy(feat_data), torch.zeros(feat_data.shape[0], dtype=torch.float32)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(id_patch),
        )

# Dataset for patch training
class ABC_patch(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch, self).__init__()
        if v_data_root is None:
            return
        self.data_root = v_data_root
        self.mode = v_training_mode
        self.conf = v_conf
        self.patch_size = 32
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.total_names = f["names"][:]
            self.ids = f["ids"][:]
            self.names = np.asarray(
                ["{:08d}_{}".format(self.total_names[i], self.ids[i]) for i in range(f["names"].shape[0])])

        self.training_id = np.arange(self.num_items)[self.total_names<800000]
        self.validation_id = np.arange(self.num_items)[np.logical_and(self.total_names<900000, self.total_names>800000)]

        self.validation_start = self.validation_id[0]

        self.is_bool_flag = self.conf["is_bool_flag"]

        assert self.resolution % self.patch_size == 0
        self.num_patch = self.resolution // self.patch_size
        self.num_patches = self.num_patch ** 2

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                self.training_id,
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                self.validation_id,
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
        else:
            raise ""
        self.index = self.index.reshape((-1, 2))

    def __len__(self):
        # return 64
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()

        ps = self.patch_size
        x_start = (v_id_patch // self.num_patch) * ps
        y_start = (v_id_patch % self.num_patch) * ps

        with h5py.File(self.data_root, "r") as f:
            if False:
                feat = np.asarray(f["features"][v_id_item], dtype=np.float32)
                flags = np.asarray(f["flags"][v_id_item], dtype=np.float32) > 0
                udf = feat[:,:,:,0:1] / 65535 * 2
                gradient = angle2vector(feat[:,:,:,1:3])
                p = (generate_coords(256) + gradient * udf).reshape(-1,3)
                export_point_cloud("1.ply", p)
                export_point_cloud("2.ply", generate_coords(256)[flags])

            features = f["features"][
                       v_id_item,
                       x_start:x_start + ps,
                       y_start:y_start + ps,
                       ].astype(np.float32)
            flags = (f["flags"][
                     v_id_item,
                     x_start:x_start + ps,
                     y_start:y_start + ps,
                     ])
            if self.is_bool_flag:
                flags = (flags>0).astype(bool).astype(np.float32)
            else:
                shifts = np.arange(26)
                flags2 = (flags[None,] & (1<<shifts)[:,None,None,None]) > 0
                flags = flags2.astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()

        features = features.reshape(ps, ps, self.num_patch, ps, -1).transpose(2, 0, 1, 3, 4)
        flags = flags.reshape(-1, ps, ps, self.num_patch, ps).transpose(3, 0, 1, 2, 4)

        times[1] += time.time() - cur_time
        return features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        feat_data, flag_data = self.get_patch(id_object, id_patch)
        return feat_data, flag_data, self.names[id_object], np.arange(flag_data.shape[0], dtype=np.int64) + id_patch * \
                                                            flag_data.shape[0]

    @staticmethod
    def collate_fn(v_batches):
        feat_data, flag_data, names, id_patch = [], [], [], []
        for item in v_batches:
            feat_data.append(item[0])
            flag_data.append(item[1])
            names.append(item[2])
            id_patch.append(item[3])
        feat_data = np.stack(feat_data, axis=0)
        flag_data = np.stack(flag_data, axis=0)
        id_patch = np.stack(id_patch, axis=0)
        names = np.asarray(names)

        return (
            (torch.from_numpy(feat_data), torch.zeros(feat_data.shape[0], dtype=torch.float32)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(id_patch),
        )

class ABC_patch2(ABC_patch):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch2, self).__init__(None,None,None)
        self.data_root = v_conf["training_root"] if v_training_mode=="training" else v_conf["validation_root"]
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]

        self.validation_start = 0

        # self.augment = v_conf["augment"]
        # self.augment = False
        self.is_bool_flag = self.conf["is_bool_flag"]
        self.mini_batch_size = self.conf["mini_batch_size"]
        self.num_total_batches = self.num_items // self.mini_batch_size

        self.index = np.arange(self.num_total_batches)

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()

        id_start = v_id_item  * self.mini_batch_size

        with h5py.File(self.data_root, "r") as f:
            if False:
                feat = np.asarray(f["features"][v_id_item], dtype=np.float32)
                flags = np.asarray(f["flags"][v_id_item], dtype=np.float32) > 0
                udf = feat[:,:,:,0:1] / 65535 * 2
                gradient = angle2vector(feat[:,:,:,1:3])
                p = (generate_coords(256) + gradient * udf).reshape(-1,3)
                export_point_cloud("1.ply", p)
                export_point_cloud("2.ply", generate_coords(256)[flags])

            features = f["features"][id_start:id_start + self.mini_batch_size,].astype(np.float32)
            flags = f["flags"][id_start:id_start + self.mini_batch_size,]
            names = f["names"][id_start:id_start + self.mini_batch_size,]
            if self.is_bool_flag:
                flags = (flags>0).astype(bool).astype(np.float32)
            else:
                shifts = np.arange(26)
                flags2 = (flags[None,] & (1<<shifts)[:,None,None,None]) > 0
                flags = flags2.astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()

        # Do this in training code
        # if self.augment and self.mode=="training":
        #     axis = np.random.randint(0,4)
        #     if axis >= 1:
        #         features = np.flip(features, axis)
        #         flags = np.flip(flags, axis)

        # features = features.transpose(0, 4, 1, 2, 3)
        flags = flags[:,:,:,:,None].transpose(0, 4, 1, 2, 3)

        times[1] += time.time() - cur_time
        return features, flags, names

    def __getitem__(self, idx):
        id_object = self.index[idx]

        feat_data, flag_data, names = self.get_patch(id_object, -1)
        return feat_data, flag_data, names, np.arange(flag_data.shape[0], dtype=np.int64)


# Whole field
class ABC_whole_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_whole_pc, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["flags"].shape[0]
            self.resolution = f["flags"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and not self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            flags = f["flags"][v_id_item].astype(bool).astype(np.float32)
            point_features = f["point_features"][v_id_item].astype(np.float32)
            coords = self.coords

        times[0] += time.time() - cur_time
        cur_time = time.time()
        return point_features[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )

################################################################################################################

# Dataset for patch training
class ABC_pc_patch(ABC_patch):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_pc_patch, self).__init__(v_data_root, v_training_mode, v_conf)

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()

        ps = self.patch_size
        x_start = (v_id_patch // self.num_patch) * ps
        y_start = (v_id_patch % self.num_patch) * ps

        with h5py.File(self.data_root, "r") as f:
            features = f["point_features"][
                       v_id_item,
                       x_start:x_start + ps,
                       y_start:y_start + ps,
                       ].astype(np.float32)
            flags = (f["flags"][
                     v_id_item,
                     x_start:x_start + ps,
                     y_start:y_start + ps,
                     ])

            times[0] += time.time() - cur_time
            cur_time = time.time()

            if self.is_bool_flag:
                flags = flags.astype(bool).astype(np.float32)[None,]
            else:
                shifts = np.arange(26)
                flags2 = (flags[None,] & (1<<shifts)[:,None,None,None]) > 0
                flags = flags2.astype(np.float32)

        times[1] += time.time() - cur_time
        cur_time = time.time()

        features = features.reshape(ps, ps, self.num_patch, ps, -1).transpose(2, 0, 1, 3, 4)
        flags = flags.reshape(-1, ps, ps, self.num_patch, ps).transpose(3, 0, 1, 2, 4)

        times[2] += time.time() - cur_time
        return features, flags


# Do overlap during the training
class ABC_patch_overlap(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch_overlap, self).__init__()
        self.data_root = v_data_root
        self.mode = v_training_mode
        self.conf = v_conf
        self.patch_size = 32
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        assert self.resolution % self.patch_size == 0
        self.num_patch = self.resolution // (self.patch_size // 2) - 1
        self.num_patches = self.num_patch ** 2

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
        else:
            raise ""
        self.index = self.index.reshape((-1, 2))

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()
        patch_size_2 = self.patch_size // 2
        patch_size_4 = patch_size_2 // 2

        x_start = v_id_patch // self.num_patch * patch_size_2
        y_start = v_id_patch % self.num_patch * patch_size_2

        with h5py.File(self.data_root, "r") as f:
            features = f["features"][
                       v_id_item,
                       x_start:x_start + self.patch_size,
                       y_start:y_start + self.patch_size,
                       ].astype(np.float32)
            # Only predict the central part of this cell
            flags = (f["flags"][
                     v_id_item,
                     x_start:x_start + self.patch_size,
                     y_start:y_start + self.patch_size,
                     ]).astype(bool).astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()
        features = np.lib.stride_tricks.sliding_window_view(
            features, window_shape=self.patch_size, axis=2
        )[:, :, ::patch_size_2].transpose(2, 0, 1, 4, 3)
        flags = np.lib.stride_tricks.sliding_window_view(
            flags, window_shape=self.patch_size, axis=2
        )[:, :, ::patch_size_2].transpose(2, 0, 1, 3)

        flags = flags[:,
                patch_size_4:-patch_size_4,
                patch_size_4:-patch_size_4,
                patch_size_4:-patch_size_4]
        times[1] += time.time() - cur_time
        return features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        feat_data, flag_data = self.get_patch(id_object, id_patch)
        return feat_data, flag_data, self.names[id_object], np.arange(flag_data.shape[0], dtype=np.int64) + id_patch * \
                                                            flag_data.shape[0]

    @staticmethod
    def collate_fn(v_batches):
        feat_data, flag_data, names, id_patch = [], [], [], []
        for item in v_batches:
            feat_data.append(item[0])
            flag_data.append(item[1])
            names.append(item[2])
            id_patch.append(item[3])
        feat_data = np.stack(feat_data, axis=0)
        flag_data = np.stack(flag_data, axis=0)
        id_patch = np.stack(id_patch, axis=0)
        names = np.asarray(names)

        return (
            torch.from_numpy(feat_data),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(id_patch),
        )


# Read pointcloud and features
class ABC_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_pc, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        self.batch_size = 64
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            names = f["names"]
            ids = f["ids"]
            self.names = np.asarray(
                ["{:08d}_{}".format(names[i], ids[i]) for i in range(ids.shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        assert self.resolution % self.batch_size == 0
        self.num_batches_per_item = ((self.resolution // self.batch_size) ** 3)

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(1), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_batches_per_item), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_batches_per_item), indexing="ij"), axis=2).reshape(-1, 2)
        else:
            raise ""

        self.coords = np.mgrid[:self.resolution, :self.resolution, :self.resolution] / (self.resolution - 1) * 2 - 1
        self.coords = np.transpose(self.coords, (1, 2, 3, 0)).astype(np.float32)

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, id_patch):
        times = [0] * 10
        cur_time = time.time()
        if self.mode == "training":
            id_item = v_id_item
            id_patch = 0
            index = np.random.randint(0, self.resolution - 1, (self.batch_size, self.batch_size, self.batch_size, 3))
            with h5py.File(self.data_root, "r") as f:
                points = f["poisson_points"][v_id_item].astype(np.float32)
                flags = f["flags"][v_id_item][index[..., 0], index[..., 1], index[..., 2]].astype(bool).astype(
                    np.float32)
                coords = self.coords[index[..., 0], index[..., 1], index[..., 2]]
        else:
            nums_per_dim = self.resolution // self.batch_size
            bs = self.batch_size
            xs = id_patch // nums_per_dim // nums_per_dim * bs
            ys = id_patch // nums_per_dim % nums_per_dim * bs
            zs = id_patch % nums_per_dim * bs

            with h5py.File(self.data_root, "r") as f:
                flags = f["flags"][v_id_item][xs:xs + bs, ys:ys + bs, zs:zs + bs].astype(bool).astype(np.float32)
                points = f["poisson_points"][v_id_item].astype(np.float32)
                coords = self.coords[xs:xs + bs, ys:ys + bs, zs:zs + bs]

        times[0] += time.time() - cur_time
        cur_time = time.time()
        times[1] += time.time() - cur_time
        return points[None, :], coords[None, :], flags[None, :], id_patch

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object, id_patch)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )

# Only read point cloud and flags
class ABC_points_and_flags(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_points_and_flags, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        self.batch_size = 64
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            names = f["names"]
            ids = f["ids"]
            self.names = np.asarray(
                ["{:08d}_{}".format(names[i], ids[i]) for i in range(ids.shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(1), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(1), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(1), indexing="ij"), axis=2).reshape(-1, 2)
        else:
            raise ""

        self.coords = np.mgrid[:self.resolution, :self.resolution, :self.resolution] / (self.resolution - 1) * 2 - 1
        self.coords = np.transpose(self.coords, (1, 2, 3, 0)).astype(np.float32)

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, id_patch):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            points = f["poisson_points"][v_id_item].astype(np.float32)
            flags = f["flags"][v_id_item].astype(bool).astype(np.float32)
            coords = self.coords

        times[0] += time.time() - cur_time
        cur_time = time.time()
        times[1] += time.time() - cur_time
        return points[None, :], coords[None, :], flags[None, :], id_patch

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object, id_patch)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )

# Dynamically generate point features
class ABC_pc_dynamic(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_pc_dynamic, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            distances = f["features"][v_id_item][:, :, :, 0].astype(np.float32) / 65535 * 2
            distance_flag = distances < 0.1
            flags = f["flags"][v_id_item].astype(bool)
            points = f["poisson_points"][v_id_item].astype(np.float32)
            coords = self.coords[distance_flag]
            flags = flags[distance_flag].astype(np.float32)

        if self.mode == "training":
            if coords.shape[0] > self.max_training_sample:
                index = np.arange(coords.shape[0])
                np.random.shuffle(index)
                coords = coords[index[:self.max_training_sample]]
                flags = flags[index[:self.max_training_sample]]

        times[0] += time.time() - cur_time
        cur_time = time.time()
        return points[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )

# Calculate the point features online using kdtree
# Not used because it is so slow
class ABC_whole_pc_dynamic(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_whole_pc_dynamic, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["flags"].shape[0]
            self.resolution = f["flags"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            flags = f["flags"][v_id_item].astype(bool).astype(np.float32)
            points = f["poisson_points"][v_id_item].astype(np.float32) / 65535 * 2 - 1
            coords = self.coords

        times[0] += time.time() - cur_time
        cur_time = time.time()

        kdtree = faiss.IndexFlatL2(3)
        res = faiss.StandardGpuResources()
        gpu_kdtree = faiss.index_cpu_to_gpu(res, 0, kdtree)

        gpu_kdtree.add(points[:, :3])
        dists, indices = gpu_kdtree.search(coords.reshape(-1, 3), 1)

        times[1] += time.time() - cur_time
        cur_time = time.time()

        udf = np.sqrt(dists).reshape(self.resolution, self.resolution, self.resolution, 1)
        gradients = points[indices.reshape(-1), :3] - coords.reshape(-1, 3)
        gradients = (gradients / np.linalg.norm(gradients, axis=1, keepdims=True)).reshape(
            self.resolution, self.resolution, self.resolution, 3)
        normals = points[indices.reshape(-1), 3:6].reshape(
            self.resolution, self.resolution, self.resolution, 3)

        point_features = np.concatenate([udf, gradients, normals], axis=-1)

        times[2] += time.time() - cur_time
        cur_time = time.time()

        return point_features[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )


def prepare_mesh(v_file, v_output_root, prefix):
    if not os.path.exists(v_file):
        print("Cannot find ", v_file)
        raise
    mesh = o3d.io.read_triangle_mesh(v_file)
    points = np.asarray(mesh.vertices)
    points = normalize_points(points)
    mesh.vertices = o3d.utility.Vector3dVector(points)
    if v_output_root is not None:
        o3d.io.write_triangle_mesh(os.path.join(v_output_root, prefix + "_norm.ply"), mesh)
    return mesh

def prepare_udf(triangles, normals, query_points, v_resolution):
    num_queries = query_points.shape[0]

    query_result = cuda_distance.query(
        triangles.reshape(-1),
        query_points.reshape(-1),
        512, 512 ** 3)

    udf = np.asarray(query_result[0]).astype(np.float32)
    closest_points = np.asarray(query_result[1]).reshape((num_queries, 3)).astype(np.float32)
    dir = closest_points - query_points
    dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)

    normals = normals[query_result[2]].astype(np.float32)

    # Revised at 1004
    feat_data = np.concatenate([udf[:, None], dir, normals], axis=1)
    feat_data = feat_data.reshape(
        (v_resolution, v_resolution, v_resolution, 7)).astype(np.float32)
    return feat_data


class ABC_test_mesh(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_resolution=256, v_output_root=None):
        super(ABC_test_mesh, self).__init__()
        self.batch_size = v_batch_size
        self.data_root = v_data_root
        self.resolution = v_resolution

        assert v_resolution % 32 == 0
        self.coords = generate_coords(v_resolution).reshape(-1, 3)
        self.num_patches = (v_resolution // 32) ** 3
        assert self.num_patches % v_resolution == 0
        prefix = Path(v_data_root).stem

        mesh = prepare_mesh(v_data_root, v_output_root, prefix)

        mesh.compute_triangle_normals()
        # v = np.asarray(mesh.vertices)
        # v[:,0]=-v[:,0]
        # mesh.vertices = o3d.utility.Vector3dVector(v)

        # UDF
        points = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        self.feat_data = prepare_udf(points[faces], normals, self.coords, v_resolution)

        self.patch_size = 32
        self.patch_list = []
        for x in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
            for y in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                for z in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                    self.patch_list.append((x, y, z))

        pass

    def __len__(self):
        return math.ceil(len(self.patch_list) / self.batch_size)

    def __getitem__(self, idx):
        features = []
        id_list = []
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            if id >= len(self.patch_list):
                break
            feat_data = self.feat_data[
                        self.patch_list[id][0]:self.patch_list[id][0] + self.patch_size,
                        self.patch_list[id][1]:self.patch_list[id][1] + self.patch_size,
                        self.patch_list[id][2]:self.patch_list[id][2] + self.patch_size,
                        ]
            features.append(np.transpose(feat_data, [3, 0, 1, 2]))
            id_list.append(self.patch_list[id])
        features = np.stack(features, axis=0)
        return features, id_list


class ABC_test_mesh_aug(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_resolution=256, v_output_root=None):
        super(ABC_test_mesh_aug, self).__init__()
        self.batch_size = v_batch_size
        self.data_root = v_data_root
        self.resolution = v_resolution

        assert v_resolution % 32 == 0
        self.coords = generate_coords(v_resolution).reshape(-1, 3)
        self.num_patches = (v_resolution // 32) ** 3
        assert self.num_patches % v_resolution == 0
        prefix = Path(v_data_root).stem

        mesh = prepare_mesh(v_data_root, v_output_root, prefix)

        # UDF1
        points = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        feat_data1 = prepare_udf(points[faces], normals, self.coords, v_resolution)

        # UDF2
        points[:,0]=-points[:,0]
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        feat_data2 = prepare_udf(points[faces], normals, self.coords, v_resolution)

        # UDF3
        points[:,0]=-points[:,0]
        points[:,1]=-points[:,1]
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        feat_data3 = prepare_udf(points[faces], normals, self.coords, v_resolution)

        # UDF4
        points[:,1]=-points[:,1]
        points[:,2]=-points[:,2]
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        feat_data4 = prepare_udf(points[faces], normals, self.coords, v_resolution)

        self.feat_data = np.stack([feat_data1, feat_data2, feat_data3, feat_data4], axis=0)

        self.patch_size = 32
        self.patch_list = []
        for i_aug in range(4):
            for x in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                for y in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                    for z in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                        self.patch_list.append((i_aug, x, y, z))
            if i_aug == 0:
                self.num_patch_per_instance = len(self.patch_list)

        pass

    def __len__(self):
        return math.ceil(len(self.patch_list) / self.batch_size)

    def __getitem__(self, idx):
        features = []
        id_list = []
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            if id >= len(self.patch_list):
                break
            feat_data = self.feat_data[
                        self.patch_list[id][0],
                        self.patch_list[id][1]:self.patch_list[id][1] + self.patch_size,
                        self.patch_list[id][2]:self.patch_list[id][2] + self.patch_size,
                        self.patch_list[id][3]:self.patch_list[id][3] + self.patch_size,
                        ]
            features.append(np.transpose(feat_data, [3, 0, 1, 2]))
            id_list.append(self.patch_list[id])
        features = np.stack(features, axis=0)
        return features, id_list


class ABC_test_voxel(torch.utils.data.Dataset):
    def __init__(self, v_conf, v_resolution=256, batch_size=128):
        super(ABC_test_voxel, self).__init__()
        self.data_root = Path(v_conf["root"])
        self.type = v_conf["type"]
        self.resolution = v_resolution
        self.batch_size=batch_size
        self.coords = generate_coords(v_resolution).reshape(-1,3)

        tasks = [self.data_root / "feat" / self.type / item for item in os.listdir(
            self.data_root / "feat" / self.type)]
        self.tasks = sorted(tasks)
        # self.tasks = self.tasks[-1:]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        prefix = self.tasks[idx].stem
        mesh_udf = np.load(
            self.tasks[idx]
        ).reshape(self.resolution, self.resolution, self.resolution, -1).astype(np.float32)

        mesh_udf = torch.from_numpy(mesh_udf).to("cuda")

        udf = de_normalize_udf((mesh_udf[..., 0:1]))
        gradients = de_normalize_angles((mesh_udf[..., 1:3]))
        normal = de_normalize_angles((mesh_udf[..., 3:5]))

        mesh_udf = torch.cat([udf, gradients, normal], dim=-1).cpu().numpy()

        sliding_data = np.lib.stride_tricks.sliding_window_view(
            mesh_udf, [32, 32, 32, 7])[::16, ::16, ::16].reshape(-1, 32, 32, 32, 7)

        num_batch = sliding_data.shape[0] // self.batch_size
        block_end = num_batch * self.batch_size
        batched_data = np.split(sliding_data[:block_end], num_batch) + [sliding_data[block_end:]]

        gt_flags = torch.tensor((0,0),dtype=torch.int64)
        # gt_flags = np.frombuffer(open(str(self.data_root / "gt" / "voronoi" / prefix), "rb").read(), dtype=np.int8)
        # gt_flags = (gt_flags[:, None] & (1 << np.arange(8))[None, :]) > 0
        # gt_flags = gt_flags.reshape(-1).reshape(256, 256, 256)

        return (
            prefix,
            batched_data,
            gt_flags,
            mesh_udf
        )
