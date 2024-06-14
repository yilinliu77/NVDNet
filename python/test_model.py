import importlib
import os.path
from multiprocessing import Process, Queue
from pathlib import Path

import hydra
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_utils import *
from abc_hdf5_dataset import generate_coords

#
# Test both mesh udf and udc udf in the testset
#

evaluate = False

def write_mesh(queue, v_output_root, v_query_points):
    precisions=[]
    recalls=[]
    f1s=[]
    while True:
        data = queue.get()
        if data is None:
            break
        final_flags, prefix, mesh_udf, gt_flags = data

        valid_mask = (mesh_udf[..., 0] < 0.2)[8:-8, 8:-8, 8:-8]
        res = mesh_udf.shape[0]
        if evaluate:
            precision = (final_flags & gt_flags[8:-8, 8:-8, 8:-8])[valid_mask].sum() / (final_flags[valid_mask].sum() + 1e-6)
            recall = (final_flags & gt_flags[8:-8, 8:-8, 8:-8])[valid_mask].sum() / (gt_flags[8:-8, 8:-8, 8:-8][valid_mask].sum()+1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1s.append(f1.cpu().numpy())
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

        final_flags = torch.nn.functional.pad(final_flags, (8, 8, 8, 8, 8, 8), mode="constant", value=1).cpu().numpy()

        final_features = mesh_udf
        valid_points = v_query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.2))]
        # valid_points = query_points[np.logical_and(gt_flags, (final_features[..., 0] < 0.4))]

        predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
        gradients_and_udf = final_features

        export_point_cloud(str(v_output_root / (prefix + ".ply")), valid_points)
        np.save(str(v_output_root / (prefix + "_feat")), gradients_and_udf)
        np.save(str(v_output_root / (prefix + "_pred")), predicted_labels)

    print("Precision: {:.4f}".format(np.nanmean(precisions)))
    print("Recall: {:.4f}".format(np.nanmean(recalls)))
    print("F1: {:.4f}".format(np.nanmean(f1s)))
    print("NAN: {:.4f}/{:.4f}".format(np.isnan(f1s).sum(), len(f1s)))
    return

@hydra.main(config_name="test_model.yaml", config_path="", version_base="1.1")
def main(v_cfg: DictConfig):
    # Predefined variables
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))
    output_dir = Path(v_cfg["dataset"]["output_dir"])

    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    ps = 32
    query_points = generate_coords(res)

    # Dataset
    batch_size = v_cfg["trainer"]["batch_size"]
    mod = importlib.import_module('abc_hdf5_dataset')
    dataset = getattr(mod, v_cfg["dataset"]["dataset_name"])(
        v_cfg["dataset"],
        res,
        batch_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=v_cfg["trainer"]["num_worker"],
        pin_memory=False,
        drop_last=False
    )
    # check_dir(Path(v_cfg["dataset"]["root"]) / "prediction" / v_cfg["dataset"]["type"])
    check_dir(output_dir)
    data_root = dataset.data_root
    type = dataset.type

    # Load model
    mod = importlib.import_module('model')
    model = getattr(mod, v_cfg["model"]["model_name"])(
        v_cfg["model"]
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    queue = Queue()
    writer = Process(target=write_mesh, args=(queue, output_dir, query_points))
    writer.start()

    # tasks = tasks[9:10]
    # Start inference
    bar = tqdm(total=len(dataloader))
    precisions=[]
    recalls=[]
    f1s=[]
    for data in dataloader:
        prefix = data[0][0]
        batched_data = [item[0] for item in data[1]]
        gt_flags = data[2][0].cuda()
        mesh_udf = data[3][0].numpy()

        bar.set_description(prefix)

        # Without augment
        if False:
            predictions = []
            for item in batched_data:
                feat = item.cuda().permute((0, 4, 1, 2, 3)).unsqueeze(1)
                feat = [(feat, torch.zeros(feat.shape[0])), None]
                prediction = model(feat, False)[0].reshape(-1, ps, ps, ps)
                predictions.append(prediction)
            predictions = torch.cat(predictions, dim=0).reshape(15,15,15,32,32,32)
            predictions = predictions[:, :, :, 8:24, 8:24, 8:24].permute(0, 3, 1, 4, 2, 5).reshape(240, 240, 240)
            predictions = torch.sigmoid(predictions)
        # With augment
        else:
            aug_predictions = []
            for i in range(4):
                predictions = []
                for item in batched_data:
                    feat = item.cuda().permute((0, 4, 1, 2, 3)).unsqueeze(1)
                    if i>=1:
                        feat[:,:,i] *= -1
                        feat = torch.flip(feat, dims=[2+i])
                    feat = [(feat, torch.zeros(feat.shape[0])), None]
                    prediction = model(feat, False)[0].reshape(-1, ps, ps, ps)
                    if i>=1:
                        prediction = torch.flip(prediction, dims=[i])
                    predictions.append(prediction)
                predictions = torch.cat(predictions, dim=0)
                num_batches_dim = round(pow(predictions.shape[0], 1/3))
                predictions = predictions.reshape(num_batches_dim, num_batches_dim,num_batches_dim, ps, ps, ps)
                dim = num_batches_dim * 16
                predictions = predictions[:, :, :, 8:24, 8:24, 8:24].permute(0, 3, 1, 4, 2, 5).reshape(dim, dim, dim)
                aug_predictions.append(torch.sigmoid(predictions))
            predictions = torch.stack(aug_predictions, dim=0).mean(dim=0)
        final_flags = predictions > threshold

        queue.put((final_flags.cpu(), prefix, mesh_udf, gt_flags.cpu()))

        # precision = (final_flags & gt_flags[8:-8,8:-8,8:-8]).sum() / final_flags.sum()
        # recall = (final_flags & gt_flags[8:-8,8:-8,8:-8]).sum() / gt_flags[8:-8,8:-8,8:-8].sum()
        # f1 = 2 * precision * recall / (precision + recall)
        # precisions.append(precision.cpu().numpy())
        # recalls.append(recall.cpu().numpy())
        # f1s.append(f1.cpu().numpy())
        #
        # final_flags = torch.nn.functional.pad(final_flags, (8,8,8,8,8,8), mode="constant", value=0).cpu().numpy()
        #
        # final_features = mesh_udf
        # valid_points = query_points[np.logical_and(final_flags, (final_features[..., 0] < 0.4))]
        # # valid_points = query_points[np.logical_and(gt_flags, (final_features[..., 0] < 0.4))]
        #
        # predicted_labels = final_flags.astype(np.ubyte).reshape(res, res, res)
        # gradients_and_udf = final_features
        #
        # export_point_cloud(str(output_dir / (prefix+".ply")), valid_points)
        # np.save(str(output_dir / (prefix+"_feat")), gradients_and_udf)
        # np.save(str(output_dir / (prefix+"_pred")), predicted_labels)
        bar.update(1)
    queue.put(None)
    writer.join()


if __name__ == '__main__':
    main()
