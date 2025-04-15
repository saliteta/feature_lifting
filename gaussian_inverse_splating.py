'''
    1. Basically, we input a Gaussian model, and a sequence of camera position
    2. We then output each alpha value and density weight accordingly
'''


from pathlib import Path
import torch
from nerfstudio.cameras.cameras import Cameras
from typing import Tuple
import torch.nn.functional as F
from tqdm import trange, tqdm
from gsplat_ext.rasterization import inverse_rasterization
import argparse
from gs_loader.dataparser import base_kernel_loader_config, general_gaussian_loader, general_cameras_loader

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def reverse_rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    image_features: torch.Tensor, #require grad
    camera: Cameras) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
    """
    Args: 
        means: Gaussian Means, in the shape of (N,3)
        quats: Gaussian quats, in the shape of (N,3)
        scales: Gaussian scales, in the shape of (N,3)
        opacities: Gaussian opacities, in the shape of (N,3)
        image_features: image feature maps, in the shape of (C, H, W, Feature Length)
    Retrus: 
        feature_gaussian: The updated Gaussian result that obtained from current feature_map
        feature_weights: The weight summary for each Gaussian, this part should be the same as the gradient of color
        gaussian_id: for get the gaussian's id for update to the original unpacked Gaussian model
    """
    viewmat = get_viewmat(camera.camera_to_worlds).cuda()

    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())

    feature_gaussian, feature_weight, infos = inverse_rasterization(
        means=means,
        quats=quats,  # rasterization does normalization internally
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        image_features=image_features,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K,  # [1, 3, 3]
        width=W,
        height=H,
        packed=True,
        near_plane=0.01,
        far_plane=1e10,
        render_mode='RGB',
        sh_degree=2,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="antialiased",
        # set some threshold to disregrad small gaussians for faster rendering.
        # radius_clip=3.0,
    )

    gaussian_ids = infos['gaussian_ids']

    return feature_gaussian, feature_weight, gaussian_ids


def frequency_filtering(scales:torch.Tensor, means: torch.Tensor, quats: torch.Tensor, opacities: torch.Tensor, threshold_size = 1e6):
    if len(opacities) <= threshold_size:
        return means,quats,scales,opacities, None
    size = scales.mean(dim=1)
    _, top_indices = torch.topk(size, k=int(threshold_size))

    means = means[top_indices]
    quats = quats[top_indices]
    scales = scales[top_indices]
    opacities = opacities[top_indices]
    return means,quats,scales,opacities, top_indices



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', help="data(colmap/scannet/nerfstudio) location of your nerfstudio initial dataset", type=str, required=True)
    parser.add_argument('--data_mode', help="colmap location of your nerfstudio initial dataset", type=str, required=False, choices=["colmap", "scannet", "nerfstudio"], default="colmap")
    parser.add_argument('--pretrained_location', help="the trained result like step-000029999.ckpt or ply file", type=str, required=True)
    parser.add_argument('--feature_location', help="a folder of pt or a folder of ftz files", type=str, required=True)
    parser.add_argument('--output_feature', help='the place u save your feature pt', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parser()

    data_location = args.data_location
    data_mode = args.data_mode
    pretrained_location = args.pretrained_location
    feature_location:str = args.feature_location
    output_features:str = args.output_feature

    feature_h, feature_w = 224, 288 # this is for the featup shape

    kernel_loader_config = base_kernel_loader_config(
        kernel_location=pretrained_location,
        kernel_type= pretrained_location.split('.')[-1]
    )

    kernel_loader = general_gaussian_loader(kernel_loader_config)
    geometry = kernel_loader.load().geometry
    scales, means, quats, opacities = geometry["scales"], geometry["means"], geometry["quats"], geometry["opacities"]
    means,quats,scales,opacities, indices = frequency_filtering(scales, means, quats, opacities, threshold_size=1e6)

    data_parser = general_cameras_loader(
        data_path=Path(data_location),
        feature_path=feature_location,
        mode=data_mode,)
    features_gaussian = torch.zeros(size = (len(opacities), 512), dtype=torch.float32).cuda()


    with torch.no_grad():
        for i in trange(len(data_parser)):
            (camera, img_names, feature_location) = data_parser[i]
            camera = camera.to('cuda')
            feature_origin: torch.Tensor = torch.load(feature_location)
            feature_origin = feature_origin.cuda()
            feature: torch.Tensor = torch.nn.functional.interpolate(feature_origin, size = (int(camera.height[0][0]), int(camera.width[0][0])), mode='bilinear', align_corners=False)
            feature = F.normalize(feature.float(), dim=1)

            del feature_origin
            torch.cuda.empty_cache()
            feature = feature.permute(0,2,3,1)
            features_gaussian_per_image, features_weight, gaussian_ids = reverse_rasterization(means, quats, scales, opacities, feature, camera)
            features_gaussian[gaussian_ids] += features_gaussian_per_image

    torch.save(features_gaussian, output_features)
    if indices != None:
        torch.save(indices, output_features.split('.')[0]+'_id.pt')