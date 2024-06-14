import os

import numpy as np
import open3d as o3d
from torchvision.ops import sigmoid_focal_loss

from abc_hdf5_dataset import generate_coords

from torch import nn
from torch.nn import functional as F
import torch
from common_utils import sigmoid, export_point_cloud

# Adopt the implementation in pytorch, but prevent NaN values
def focal_loss(inputs, targets, v_alpha=0.75, gamma: float = 2, ):
    loss = sigmoid_focal_loss(inputs, targets,
                              alpha=v_alpha,
                              reduction="mean"
                              )

    # p = torch.sigmoid(inputs.to(torch.float32))
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    #
    # if v_alpha >= 0:
    #     alpha_t = v_alpha * targets + (1 - v_alpha) * (1 - targets)
    #     loss = alpha_t * loss
    #
    # Check reduction option and return loss accordingly
    # loss = loss.mean()
    return loss


def BCE_loss(v_predictions, labels, v_alpha=0.75):
    loss = nn.functional.binary_cross_entropy_with_logits(v_predictions, labels,
                                                          reduction="mean"
                                                          )
    return loss


class Residual_fc(nn.Module):
    def __init__(self, v_input, v_output):
        super().__init__()
        self.fc = nn.Linear(v_input, v_output)
        self.relu = nn.LeakyReLU()

    def forward(self, v_data):
        feature = self.relu(self.fc(v_data))
        return feature + v_data


def de_normalize_angles(v_angles):
    angles = (v_angles / 65535 * torch.pi * 2)
    dx = torch.cos(angles[..., 0]) * torch.sin(angles[..., 1])
    dy = torch.sin(angles[..., 0]) * torch.sin(angles[..., 1])
    dz = torch.cos(angles[..., 1])
    gradients = torch.stack([dx, dy, dz], dim=-1)
    return gradients


def de_normalize_udf(v_udf):
    return v_udf / 65535 * 2


def de_normalize_points(v_points):
    return v_points / 65535 * 2 - 1


#################################################################################################################
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, with_bn=True, kernel_size=3, padding=1, stride=1, dilate=1):
        super(conv_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,
                      dilation=dilate),
            nn.BatchNorm3d(ch_out) if with_bn else nn.Identity(),
            # nn.InstanceNorm3d(ch_out) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True, dilation=1),
            nn.BatchNorm3d(ch_out) if with_bn else nn.Identity(),
            # nn.InstanceNorm3d(ch_out) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x).contiguous()
        return x


class U_Net_3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, v_depth=5, v_pool_first=True, base_channel=16, with_bn=True):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.depths = v_depth

        self.conv = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        if v_pool_first:
            self.conv1 = nn.Sequential(
                conv_block(ch_in=img_ch, ch_out=base_channel, with_bn=with_bn),
                nn.MaxPool3d(kernel_size=4, stride=4),
                conv_block(ch_in=base_channel, ch_out=base_channel, with_bn=with_bn),
            )
        elif with_bn:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(base_channel),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        cur_channel = base_channel
        for i in range(v_depth):
            self.conv.append(conv_block(ch_in=cur_channel, ch_out=cur_channel * 2, with_bn=with_bn))

            self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
            self.up_conv.append(conv_block(ch_in=cur_channel * 2, ch_out=cur_channel, with_bn=with_bn))

            cur_channel = cur_channel * 2

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, v_input):
        # encoding path
        x1 = self.conv1(v_input)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 1, -1, -1):
            item = self.up[i](up_x[-1])
            if i >= 0:
                item = torch.cat((item, x[i]), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class Base_model(nn.Module):
    def __init__(self, v_conf):
        super(Base_model, self).__init__()
        self.need_normalize = v_conf["need_normalize"]
        self.loss_func = focal_loss
        self.augment = v_conf["augment"]
        self.loss_alpha = v_conf["focal_alpha"]
        self.num_features = v_conf["channels"]
        self.ic = v_conf["channels"]  # input_channels
        self.output_c = v_conf["output_channels"]

    def loss(self, v_predictions, v_input):
        predict_labels = v_predictions[0]
        gt_labels = v_predictions[1]

        loss = self.loss_func(predict_labels, gt_labels, self.loss_alpha)
        return {"total_loss": loss}

    def compute_pr(self, v_pred, v_gt):
        v_pred = v_pred[0]
        bs = v_pred.shape[0]
        prob = torch.sigmoid(v_pred).reshape(bs, -1)
        gt = v_gt.reshape(bs, -1).to(torch.long)
        return prob, gt

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
        return
        assert gathered_prediction.shape[0] == 64
        v_resolution = 256
        query_points = generate_coords(v_resolution).reshape(-1, 3)

        predicted_labels = gathered_prediction.reshape(
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        predicted_labels = predicted_labels.max(axis=-1)
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = gt_labels > 0.5
        gt_labels = gt_labels.max(axis=-1)
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return


class Base_model_UNet(Base_model):
    def __init__(self, v_conf):
        super(Base_model_UNet, self).__init__(v_conf)
        self.encoder = U_Net_3D(
            img_ch=v_conf["channels"],
            output_ch=self.output_c,
            v_pool_first=False,
            v_depth=4,
            base_channel=v_conf["base_channel"],
            with_bn=v_conf["with_bn"]
        )

    def forward(self, v_data, v_training=False):
        (feat_data, _), flag = v_data
        bs = feat_data.shape[0]
        num_mini_batch = feat_data.shape[1]
        feat_data = feat_data.reshape((bs * num_mini_batch,) + feat_data.shape[2:])
        if flag is not None:
            flag = flag.reshape((bs * num_mini_batch,) + flag.shape[2:])

        if self.need_normalize:
            udf = de_normalize_udf(feat_data[..., 0:1])
            if self.ic == 7:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                normal = de_normalize_angles(feat_data[..., 3:5])
                x = torch.cat([udf, gradients, normal], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            elif self.ic == 4:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                x = torch.cat([udf, gradients], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            else:
                x = udf.permute((0, 4, 1, 2, 3)).contiguous()
        else:
            x = feat_data[:, :self.num_features]

        if self.augment and v_training:
            def visualize_it(v_x, v_flag):
                i = 15
                p = torch.stack(
                    torch.meshgrid(torch.arange(0, 32), torch.arange(0, 32), torch.arange(0, 32), indexing="ij"),
                    dim=-1)
                p = p.to(x.device) / 255 * 2 - 1
                dir = v_x[i, 1:4].permute(1, 2, 3, 0)
                pp = p + dir * v_x[i, 0:1].permute(1, 2, 3, 0)

                voronoi_edges = p[v_flag[i, 0] > 0]
                import open3d as o3d
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pp.reshape(-1, 3).cpu().numpy()))
                o3d.io.write_point_cloud("debug1.ply", pcd)

                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voronoi_edges.cpu().numpy()))
                o3d.io.write_point_cloud("debug2.ply", pcd)

            # visualize_it(x, flag)

            axis = np.random.randint(0, 4)
            if axis >= 1:
                x = torch.flip(x, dims=[axis+1])
                flag = torch.flip(flag, dims=[axis+1])
                x[:,axis] *= -1
            # visualize_it(new_x, new_flag)

        # Debug
        if False:
            points = (np.arange(256) / 255) * 2 - 1
            points = np.stack(np.meshgrid(points, points, points, indexing='ij'), axis=-1)
            points = points[:32, :32, :]
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            udf = x[:, 0:1].permute(2,3,0,4,1).reshape(32,32,256,1).detach().cpu().numpy()
            g = x[:, 1:4].permute(2,3,0,4,1).reshape(32,32,256,3).detach().cpu().numpy()
            n = x[:, 4:7].permute(2,3,0,4,1).reshape(32,32,256,3).detach().cpu().numpy()
            p = points + udf * g
            pcd.points = o3d.utility.Vector3dVector(p.reshape(-1,3))
            pcd.normals = o3d.utility.Vector3dVector(n.reshape(-1,3))
            o3d.io.write_point_cloud("debug.ply", pcd)

        prediction = self.encoder(x)

        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:]),\
                flag.reshape((bs, num_mini_batch,) + flag.shape[1:]) if flag is not None else None

