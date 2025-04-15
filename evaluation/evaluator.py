"""
    This evaluator can export several things:
    1. Load the model
    2. Load the evaluation image, and do the rendering
    3. rendered result are the following: 
        - Feature 
        - Feature PCA
        - Feature with text query
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Tuple
from gs_loader import dataparser, kernel_loader, feature_mapper
from nerfstudio.cameras.cameras import Cameras
from feature_viewer.renderer import renderer
import torch.nn.functional as F

import torch
import numpy as np
import cv2

from sklearn.decomposition import PCA
from tqdm import tqdm

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


class base_evaluator(ABC):
    def __init__(self, kernels_loader: kernel_loader, 
                feature_mapper: feature_mapper.feature_lift_mapper,
                dataset_path: Path, gt_paths:Path, 
                dataset_mode: Literal["colmap", "nerfstudio", "scannet"] = 'colmap') -> None:
        super().__init__()
        assert dataset_path.exists(), f"The dataset_path scanner/colmap/nerfstudio location does not exist: {dataset_path}"
        self.dataset_path = dataset_path 
        assert gt_paths.exists(), f"The gt_path location does not exist: {gt_paths}"
        self.gt_paths = gt_paths

        self.camera_loader = dataparser.general_cameras_loader(dataset_path, "temp", mode=dataset_mode)
        # No need to load feature, using fake one

        self.renderer = renderer(
            kernels_loader,
            feature_mapper
        ) # render the result

    
    @abstractmethod
    def _load_camera(self) -> Tuple[Cameras, List[Path]]:
        """
            We should process some files, and output a series of camera position
        """

    def load_camera(self) -> Tuple[Cameras, List[Path]]:
        """
            Load the evaluate camera pose into a sequence of Nerfstudio Camera
            Load a list of path, camera path or gt images path
        """

        self.cameras, self.gt_image_path = self._load_camera()

        return self.cameras, self.gt_image_path

    @abstractmethod
    def _eval(self, 
        mode: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA"]) -> List[torch.Tensor]:
        """
        Return a List of Tensors in cpu already detached
        It should support three mode, RGB, RGB+FEATURE, RGB+FEATURE+FEATURE_PCA
        the genrated result should be put inside a list, each result should be a torch tensor

        For example the last mode we have 
        [torch.shape(HW3), torch.shape(HWC), torch.shape(HW3)]
        """

    def eval(self, saving_path: Path, 
        mode: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA"], 
        feature_saving_mode = Literal['pt', 'ftz']) -> None:
        """
            We do not consider using different texts to query an attention map
            Since it can be done using our image rendering tool, or post processing
            using 2D image metrics
        """
        print(f"==== Input mode {mode}, Processing ====")
        results: List[torch.Tensor] = self._eval(mode=mode)
        
        print(f"==== Evaluation Accomslihed, Saving Results ... ====")
        saving_path.mkdir(exist_ok=True)

        if mode == "RGB":
            
            RGB_path = saving_path / "RGB"
            RGB_path.mkdir(exist_ok=True)
            images = results[0]

            for img, img_name in zip(images, self.gt_image_path):
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(RGB_path/img_name.name, img)
            
        
        elif mode == "RGB+Feature":
            
            RGB_path = saving_path / "RGB"
            RGB_path.mkdir(exist_ok=True)
            Feature_path = saving_path / "Feature"
            Feature_path.mkdir(exist_ok=True)
            images = results[0]
            features = results[1]

            for img, feature, img_name in zip(images, features, self.gt_image_path):
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(RGB_path/img_name.name, img)
                if feature_saving_mode == 'pt':
                    torch.save(feature, Feature_path/(img_name.name.split('.')[0]+'.pt'))
                elif feature_saving_mode == 'ftz':
                    print("FTZ FILE DETECTED Ask Yihan Fang")
                    raise NotImplementedError
                else:
                    print(f"currently only support saving feature in ftz and pt format, get: {feature_saving_mode}")
                    raise NotImplementedError


        elif mode == "RGB+Feature+Feature_PCA":
            
            RGB_path = saving_path / "RGB"
            RGB_path.mkdir(exist_ok=True)
            Feature_path = saving_path / "Feature"
            Feature_path.mkdir(exist_ok=True)
            Feature_PCA_path = saving_path / "Feature_PCA"
            Feature_PCA_path.mkdir(exist_ok=True)

            images = results[0]
            features = results[1]
            features_PCA = results[2]

            for img, feature, feature_pca, img_name in zip(images, features, features_PCA, self.gt_image_path):
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(RGB_path/img_name.name, img)

                torch.save(feature, Feature_path/(img_name.name.split('.')[0]+'.pt'))

                feature_pca = feature_pca.numpy()
                feature_pca = np.clip(feature_pca, 0.0, 1.0)
                feature_pca = (feature_pca * 255).astype(np.uint8)
                feature_pca = cv2.cvtColor(feature_pca, cv2.COLOR_RGB2BGR)
                cv2.imwrite(Feature_PCA_path/img_name.name, feature_pca)
        
        else:
            print(f"We only support following three mode: RGB, RGB+Feature, \
                RGB+Feature+Feature_PCA, but get {mode}")
            raise NotImplementedError
        
            
        print(f"=== evaluation succssful :), saved at:  {saving_path} ===")


class lerf_evaluator(base_evaluator):
    def __init__(self, kernels_loader: kernel_loader, 
    feature_mapper: feature_mapper.feature_lift_mapper, 
    dataset_path: Path, gt_paths: Path, dataset_mode: Literal["colmap", "nerfstudio", "scannet"] = 'colmap') -> None:
        super().__init__(kernels_loader, feature_mapper, dataset_path, gt_paths, dataset_mode)

    
    def _load_camera(self) -> Tuple[Cameras, List[Path]]:

        labels_location = self.gt_paths
        image_names:List[Path] = []
        names = [f for f in labels_location.iterdir() if f.is_file()]
        names.sort()
        for name in names:
            if name.name.endswith('jpg'):
                image_names.append(Path(name))
        
        # already get the target image and json files name, now, need to know the 
        # camera pose
        cameras = []
        count = 0

        for i in range(len(self.camera_loader)):
            camera, img_name, _ = self.camera_loader[i]
            for target_name in image_names:
                if Path(img_name).name == target_name.name: # same image
                    cameras.append(camera)
                    continue

        return cameras, image_names
    
    def _eval(self, mode: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA"]) -> List[torch.Tensor]:
        if mode == "RGB":
            results = self.rgb_render()
            return [torch.stack(results)]
        
        elif mode == "RGB+Feature":
            results = []
            results.append(torch.stack(self.rgb_render()))
            results.append(torch.stack(self.feature_render()))
        
        else:
            results = []
            results.append(torch.stack(self.rgb_render()))
            results.append(torch.stack(self.feature_render()))
            feature_pca = []
            for feature in tqdm(results[1], desc='feature pca eval'):
                feature: np.ndarray = feature.numpy()
                H,W,C = feature.shape
                flat_feat = feature.reshape(-1, C)
                pca = PCA(n_components=3)
                reduced_feat = pca.fit_transform(flat_feat)
                min_val = reduced_feat.min()
                max_val = reduced_feat.max()
                flat_features_norm = (reduced_feat - min_val) / (max_val - min_val)

                image_3d = flat_features_norm.reshape(H, W, 3)
                feature_pca.append(image_3d)
            feature_pca = torch.tensor(feature_pca)

            results.append(feature_pca)
        
        return results



    def rgb_render(self) -> List[torch.Tensor]:
        results = []
        for camera in tqdm(self.cameras, desc="RGB eval"):
            K, W, H, viewmat = self.camera_extractor(camera=camera)
            img = self.renderer.render(
                w2c=viewmat,
                k=K,
                mode = 'RGB',
                H = H, W = W
            )
            results.append(img.detach().cpu())
        return results
    
    def feature_render(self) -> List[torch.Tensor]:
        results = []
        for camera in tqdm(self.cameras, desc='feature eval'):
            K, W, H, viewmat = self.camera_extractor(camera=camera)
            img = self.renderer.render(
                w2c=viewmat,
                k=K,
                mode = 'Feature',
                H = H, W = W
            )
            C = img.shape[-1]
            feature = img.reshape((-1, C))
            feature = F.normalize(feature, dim=1)
            feature = feature.reshape((H,W,C)).detach().cpu()
            results.append(feature)

        return results



    def camera_extractor(self, camera:Cameras):
        K = camera.get_intrinsics_matrices().cuda().squeeze()
        W, H = int(camera.width.item()), int(camera.height.item())
        viewmat = get_viewmat(camera.camera_to_worlds).cuda().squeeze()
        return K, W, H, viewmat


