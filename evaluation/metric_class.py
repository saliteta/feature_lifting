
from typing import Dict, List, Tuple
import numpy as np
import torch
import os
import json

import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import pandas as pd

"""
    Here we try to encorperate several several baseline here
    Base Line 1 is using the background words 
    Base Line 2 is using the YOLO to help identify the background
    Final Version would be using ChatGPY 4o as an API or human user to do the input and output identification
    The final version will be implemented in our UI
"""

BACKGROUND_WORDS = ['floor', 'walls', 'ceiling', 'desk', 'background', 'road', 'sky']
SUPPORTED_MODEL = {'maskclip'}
SUPPORTED_BACKGROUND_FINDING = {'None', 'Light', 'Heavy', 'Human'} 





class Metrics:
    def __init__(self, label_folder, features_folder, text_encoder: str = 'maskclip') -> None:
        '''
            In Future, we need to load the rendered image, now, we just use the real image as a substituion
        '''
        self.label_folder = label_folder
        self.features_folder = features_folder
        self.background_words = BACKGROUND_WORDS
        
        self.meta_data = self.data_parser()
        self.load_model(name = text_encoder)
        self.background_detection_model = None
        self.similar_text_filtering_model = None


    def data_parser(self) -> List[Dict]:
        scene_names: List[str] = os.listdir(self.features_folder)
        scene_names.sort()

        # Data Structure: Scene Dict
        # {Scene_Name, Scene_Jsons, Scene_Images, Scene_Features}

        datasets = []

        for scene in scene_names:
            scene_dict = {}
            scene_dict['scene_name'] = scene
            scene_dict['feature_folder'] = self.features_folder
            scene_dict['labels_folder'] = self.label_folder
            # query by  scene_dict['feature_folder']/scene_dict['scene_name']/scene_jsons[i]
            scene_views_feature = os.listdir(os.path.join(self.features_folder, scene, 'Feature')) # npy files
            scene_views_feature.sort()
            scene_jsons = []
            scene_img = []
            for scene_view in scene_views_feature:
                scene_jsons.append(scene_view.split('.')[0]+'.json')
                scene_img.append(scene_view.split('.')[0]+'.jpg')
            scene_dict['features'] = scene_views_feature
            scene_dict['json'] = scene_jsons
            scene_dict['img'] = scene_img
            datasets.append(scene_dict)

        return datasets



    def json_parser(self, json_dict) -> Tuple[torch.Tensor, List[str]]:
        """
        Parse a JSON dictionary containing segmentation annotations and return:
          - A torch.Tensor of shape (num_categories, H, W) with binary masks aggregated by category (1 where any instance of the object is present, 0 elsewhere)
          - A list of unique category names.

        Args:
            json_dict (dict): The JSON dictionary with keys "info" (containing image width and height) and "objects" 
                              (each with "category" and "segmentation").

        Returns:
            Tuple[torch.Tensor, List[str]]: A tuple where the first element is a tensor of aggregated binary masks and 
                                             the second element is a list of unique category names.
        """
        import numpy as np
        import cv2
        import torch

        # Dictionary to store the aggregated mask for each category
        aggregated_masks = {}
        # List to keep track of category order
        unique_categories = []

        W = json_dict['info']['width']
        H = json_dict['info']['height']

        for obj in json_dict['objects']:
            category = obj["category"]
            seg = obj['segmentation']

            # Create a binary mask with shape (H, W)
            mask = np.zeros((H, W), dtype=np.uint8)

            # Convert polygon points to a numpy array of shape (num_points, 1, 2)
            polygon_points = np.array([[point[0], point[1]] for point in seg], dtype=np.int32).reshape((-1, 1, 2))

            # Fill the polygon on the mask (the region inside becomes 1)
            cv2.fillPoly(mask, [polygon_points], 1)

            if category in aggregated_masks:
                # If the category already exists, aggregate the new mask with the existing one.
                # Using np.maximum will combine the masks so that if either is 1, the result is 1.
                aggregated_masks[category] = np.maximum(aggregated_masks[category], mask)
            else:
                # New category: store its mask and record the category order.
                aggregated_masks[category] = mask
                unique_categories.append(category)

        # Stack masks into a single torch.Tensor with shape (num_categories, H, W)
        masks_tensor = torch.from_numpy(np.stack([aggregated_masks[cat] for cat in unique_categories]))

        return masks_tensor, unique_categories

    def view_parser(self, scene_name:str, feature_name:str)->Tuple[torch.Tensor, torch.Tensor, str, np.ndarray, List[str]]:
        """
            Per view, we input the scene name and the feature name and return
            feature_map, gt masks, image_path, and np array in cv2 rgb format image
        """
        feature_path = os.path.join(self.features_folder, scene_name, 'Feature', feature_name) 
        if feature_path.endswith('npy'):
            feature_map = torch.tensor(np.load(feature_path)).cuda()
        else:
            feature_map :torch.Tensor= torch.load(feature_path).cuda().squeeze().permute((1,2,0))

        # parse json
        json_path = os.path.join(self.label_folder, scene_name, feature_name.split('.')[0]+'.json') 
        with open(json_path, "r") as file:
            json_dict = json.load(file)
        gt_masks, texts = self.json_parser(json_dict)

        # read image
        img_path = os.path.join(self.label_folder, scene_name, feature_name.split('.')[0]+'.jpg')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Reads image in BGR by default
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return feature_map, gt_masks, img_path, image_rgb, texts

    def load_model(self, name: str):
        assert name in SUPPORTED_MODEL, f"Currently we only support the following model: {SUPPORTED_MODEL}, your model {name} is not in the list"

        if name == 'maskclip':
            from featup.featurizers.maskclip.clip import tokenize  # Maskclip tokenizer
            model = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).cuda()
            self.tokenizer = tokenize
            self.model = model
            print(f"----model {name} succssfully loaded----")

    
    def background2text(self, image: str, mode: str):
        """
            Args: Given Current Image, we need to find its background words, either using 
            predefined background words
        """
        assert mode in SUPPORTED_BACKGROUND_FINDING, f"currently we only support the following mode {SUPPORTED_BACKGROUND_FINDING}\
                None corresponds to defined background words shown up here, light corresponds to using YOLO, HEAVY using ChatGPT API, \
                and Human should using our UI instead of current scripts. But user input is {mode}"

        if mode == 'None':
            self.background_words = BACKGROUND_WORDS
        elif mode == 'Light':
            self.background_words = self.light_background_detection(image) + BACKGROUND_WORDS
        elif mode == 'Heavy':
            self.background_words = self.heavy_background_detection(image)
        else: 
            print('d Human should using our UI instead of current scripts')
            self.human_background_detection(image)
    
    def light_background_detection(self, image: str) -> List[str]:
        if self.background_detection_model == None:
            print("we are using light weight detection model for the first time, loading model")

            # Load a model
            self.background_detection_model = YOLO("yolo11n.pt")

        # Perform object detection on an image
        results = self.background_detection_model(image)
        detection: ultralytics.engine.results.Boxes = results[0]
        sequence_number = torch.unique(detection.boxes.cls)
        background_words = []
        for key in sequence_number:
            background_words.append(detection.names[int(key)])
        return background_words

    def heavy_background_detection(self, image:str) -> List[str]:
        raise NotImplementedError

    def human_background_detection(self, image:str) -> List[str]:
        raise NotImplementedError

    def text_features_generation(self, texts, filtering = True) -> torch.Tensor:
        text_features = []
        for text in texts:
            token = self.tokenizer(text).to('cuda')
            text_feature = self.model.model.model.encode_text(token).float().squeeze()
            text_features.append(text_feature)
        for text in self.background_words:
            token = self.tokenizer(text).to('cuda')
            text_feature = self.model.model.model.encode_text(token).float().squeeze()
            text_features.append(text_feature)
        
        text_features = torch.stack(text_features)  # Shape: (num_classes, D)
        text_features = text_features / torch.norm(text_features, p=2, dim=1, keepdim=True)


        if filtering:
            return self.filtering(texts, text_features)

        return text_features
        
    def load_model_open_clip(self):
        print("detect light weight detection, loading model for first iteration ...")
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
        model.eval().cuda()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print('----model clip text encoder succssfully loaded----')
        return model, tokenizer


    def filtering(self, texts, text_features):
        
        if self.similar_text_filtering_model == None: # first time detection
            self.similar_text_filtering_model = self.load_model_open_clip()

        # Separate positive and negative text features
        positive_token = self.similar_text_filtering_model[1](texts).to('cuda')
        negative_token = self.similar_text_filtering_model[1](self.background_words).to('cuda')

        positive_features = self.similar_text_filtering_model[0].encode_text(positive_token)
        negative_features = self.similar_text_filtering_model[0].encode_text(negative_token)


        # Calculate attention scores between all positive-negative pairs
        attention_scores = torch.matmul(positive_features, negative_features.T)

        # Find the maximum attention score for each positive text
        max_attention_scores, max_indices = torch.max(attention_scores, dim=0)

        positive_features = text_features[:len(texts)]
        negative_features = text_features[len(texts):]

        # Remove negative pairs where max attention score is larger than 0.8
        valid_negatives = []
        for i in range(len(negative_features)):
            if (max_attention_scores[i] < 0.8):
                valid_negatives.append(negative_features[i])

        # Stack the valid negatives
        valid_negatives = torch.stack(valid_negatives)
        valid_negatives = torch.concat((positive_features, valid_negatives))


        return valid_negatives

    def attention_calculation(self, text_features:torch.Tensor, feature_map:torch.Tensor, comparison: bool = True, top_n = 20) -> Tuple[np.ndarray, np.ndarray]:

        if comparison:
            # Compute similarity scores (cosine similarity)
            feature_map_reshaped = feature_map.view(-1, feature_map.shape[-1])  # Flatten spatial dims
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize text features
            feature_map_reshaped = feature_map_reshaped / feature_map_reshaped.norm(dim=-1, keepdim=True)  # Normalize feature map

            # Compute similarity (logits)
            attention_scores = torch.matmul(feature_map_reshaped, text_features.T)  # Shape: (H*W, num_classes)
            predicted_labels = torch.argmax(attention_scores, dim=-1)  # Get highest similarity category index


            H, W, _ = feature_map.shape
            segmentation_masks = predicted_labels.view(H, W).cpu().numpy()



        else:
            # Compute similarity scores (cosine similarity)
            feature_map_reshaped = feature_map.view(-1, feature_map.shape[-1])  # Flatten spatial dims
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize text features
            feature_map_reshaped = feature_map_reshaped / feature_map_reshaped.norm(dim=-1, keepdim=True)  # Normalize feature map

            # Compute similarity (logits)
            attention_scores = torch.matmul(feature_map_reshaped, text_features.T)  # Shape: (H*W, num_classes)

            predicted_labels = attention_scores>attention_scores.mean()
            H, W, _ = feature_map.shape
            predicted_labels = predicted_labels.transpose(0, 1)
            segmentation_masks = predicted_labels.view(-1, H, W).cpu().numpy()
        num_classes = text_features.shape[0]
        max_coords = []
        for cls in range(num_classes):
            class_attention = attention_scores[:, cls]  # (H*W,)
            # Get the top_n indices for the current class (sorted in descending order)
            topk = torch.topk(class_attention, k=top_n, sorted=True)
            coords = []
            for idx in topk.indices:
                y, x = divmod(idx.item(), W)  # Convert flat idx to 2D coord (y, x)
                coords.append((x, y))
            max_coords.append(coords)

        return segmentation_masks, np.array(max_coords)
        

    def mIou(self, predicted_dense_masks, gt_bindary_masks):
        """
        Calculate the IoU for each class between the predicted dense segmentation and ground truth binary masks,
        and compute a weighted overall mIoU per frame based on the total number of pixels in each ground truth mask.

        Args:
            predicted_dense_masks (np.ndarray): 2D array where each pixel's value represents its predicted class label.
            gt_bindary_masks (List[np.ndarray]): List of binary masks (each of shape (H, W)) for each ground truth class.

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - A list of IoU values for each class.
                - The weighted overall mIoU for the frame.
        """

        target_shape = gt_bindary_masks[0].shape  # (height, width)

        # Check if resizing is needed
        if predicted_dense_masks.shape != target_shape:
            # cv2.resize expects size as (width, height) and uses nearest neighbor to avoid smoothing
            predicted_dense_masks = cv2.resize(predicted_dense_masks, 
                                               dsize=(target_shape[1], target_shape[0]),
                                               interpolation=cv2.INTER_NEAREST)


        IoUs = [] 
        weights = []

        # Loop through each ground truth mask (each corresponding to a class)
        for i in range(len(gt_bindary_masks)):
            # Create a binary mask for predicted segmentation for class i
            predicted_mask = (predicted_dense_masks == i)
            # Get the corresponding ground truth mask for class i
            gt_mask = gt_bindary_masks[i]

            # Compute intersection and union for IoU calculation
            intersection = np.logical_and(predicted_mask, gt_mask).sum()
            union = np.logical_or(predicted_mask, gt_mask).sum()

            # Compute IoU and avoid division by zero
            iou = intersection / union if union > 0 else 0
            IoUs.append(iou)

            # Weight for the class: total number of pixels in the ground truth mask
            weights.append(gt_mask.sum())

        # Calculate overall mIoU as a weighted average using the ground truth pixel counts as weights
        total_weight = sum(weights)
        overall_miou = (sum(iou * w for iou, w in zip(IoUs, weights)) / total_weight) if total_weight > 0 else 0

        # Print overall mIoU for the frame
        # print(f"Overall mIoU for the frame: {overall_miou:.4f}")

        return IoUs, overall_miou

    def localization(self, orig_shape, max_coordinates, gt_bindary_masks):
        """
        Calculate the localization accuracy for each class by checking if any of the top n coordinates 
        is inside the corresponding ground truth binary mask. Adjust the max_coordinates to match the 
        resolution of the ground truth masks if needed, and compute the mean accuracy across classes.

        Args:
            orig_shape (tuple): The original shape (height, width) of the predicted mask.
            max_coordinates (List[List[tuple]]): List of lists of (x, y) coordinates for each class, 
                where each sublist is sorted with the highest attention scores first.
            gt_bindary_masks (List[np.ndarray]): List of binary masks (each of shape (H, W)) for each ground truth class.
            n (int): The number of top coordinates to consider for each class.

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - A list of localization accuracy values for each class (1 if any of the top n coordinates is within 
                  the ground truth mask, else 0).
                - The mean localization accuracy across classes.
        """
        target_shape = gt_bindary_masks[0].shape  # (height, width)

        # Adjust the coordinates if the original shape differs from target shape.
        if orig_shape != target_shape:
            scale_x = target_shape[1] / orig_shape[1]
            scale_y = target_shape[0] / orig_shape[0]
            adjusted_coordinates = []
            for coords in max_coordinates:
                # Take only the top n coordinates and adjust them.
                adjusted = [(int(round(x * scale_x)), int(round(y * scale_y))) for (x, y) in coords]
                adjusted_coordinates.append(adjusted)
            max_coordinates = adjusted_coordinates

        localization_acc = []
        H, W = target_shape

        # Iterate over each class (only up to the number of available ground truth masks).
        for i, coords in enumerate(max_coordinates[:len(gt_bindary_masks)]):
            found = False
            for (x, y) in coords:
                # Check that the coordinate is within bounds.
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                # If any coordinate is inside the ground truth mask (i.e. nonzero), mark as correct.
                if gt_bindary_masks[i][y, x]:
                    found = True
                    break
            localization_acc.append(1.0 if found else 0.0)

        mean_acc = sum(localization_acc) / len(localization_acc) if localization_acc else 0.0

        return localization_acc, mean_acc

    def visualization_result(self, segmentation_mask, original_images, store_location, text_disctiption) -> torch.Tensor:
        """
        Align the original image and the colored segmentation mask side by side with a legend at the bottom,
        and store the resulting figure at the specified store_location. No axes or titles are displayed.
        
        Args:
            segmentation_mask (np.ndarray): A 2D array (H x W) where each pixel is a category label.
            original_images (np.ndarray): The original image as an RGB array (H x W x 3).
            store_location (str): The file path where the resulting visualization will be saved.
            text_disctiption (list): A list of category names corresponding to the integer labels in segmentation_mask.
            
        Returns:
            torch.Tensor: A tensor representation of the saved visualization image, shape (3, H, W).
        """
        H, W = segmentation_mask.shape
        num_classes = len(text_disctiption)
        colors = plt.cm.get_cmap("tab20", num_classes).colors  # Get RGBA colors
    
        # Create colored segmentation mask
        segmentation_colored = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(num_classes):
            segmentation_colored[segmentation_mask == i] = (np.array(colors[i][:3]) * 255)
    
        # Create figure with two side-by-side subplots (no axes/titles)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={'wspace': 0, 'hspace': 0})
        ax1.imshow(original_images)
        ax1.axis("off")
        
        ax2.imshow(segmentation_colored)
        ax2.axis("off")
    
        # Create legend patches and add to figure
        legend_patches = [plt.Rectangle((0,0), 1, 1, fc=colors[i][:3]) for i in range(num_classes)]
        fig.legend(legend_patches, text_disctiption, 
                   loc='lower center', ncol=4, frameon=True, fontsize=10)
    
        # Adjust layout to make space for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Increase bottom margin for legend
    
        # Save and convert to tensor
        plt.savefig(store_location, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        
        saved_img = Image.open(store_location).convert("RGB")
        return torch.tensor(np.array(saved_img)).permute(2, 0, 1).float()


    def metrics(self, model: str = 'maskclip', mode:str = 'None', output_location:str = './output', filtering_enable = True):
        """
            metrics itself, using default text encoder maskclip, 
            default background finding mode, None
            and output the segmented mask and mIoU to correct loation
        """

        self.metrics_report = []

        print(f"using model: {model}, background detection mode: {mode}. The result will be save at location: {output_location}")

        for scene in tqdm(self.meta_data):
            per_frame_ious = []
            per_frame_accs = []
            scene_name = scene['scene_name']
            for i in range(len(scene['features'])):

                feature_name = scene['features'][i]

                feature_map, gt_masks, img_path, image_rgb, texts = self.view_parser(scene_name, feature_name)

                self.background2text(image = img_path, mode=mode)

                text_features = self.text_features_generation(texts=texts, 
                    filtering=(mode=='None' or mode == 'Light') and filtering_enable)


                segmentation_dense_masks, max_location = self.attention_calculation(text_features=text_features, feature_map = feature_map)


                os.makedirs(os.path.join(output_location, scene_name), exist_ok=True)
                store_location = os.path.join(output_location,scene_name, feature_name.split('.')[0] + '.jpg')
                self.visualization_result(segmentation_dense_masks, image_rgb,store_location, texts)

                miou_values, mIou_perframe = self.mIou(segmentation_dense_masks, gt_masks)
                acc_values, acc_perframe = self.localization(segmentation_dense_masks.shape, max_coordinates=max_location, gt_bindary_masks=gt_masks)

                per_frame_ious.append(mIou_perframe)
                per_frame_accs.append(acc_perframe)

                frame_id = feature_name[i].split('.')[0]
                for cat, iou_val, acc_val in zip(texts, miou_values, acc_values):
                   self.metrics_report.append({
                        "scene": scene_name,
                        "frame": frame_id,
                        "category": cat,
                        "iou": iou_val,
                        "acc": acc_val,
                    })
            print(f'{scene_name} mIoU is: {torch.tensor(per_frame_ious).mean()}, mAcc is {torch.tensor(per_frame_accs).mean()}')
        
        self.metrics_report = pd.DataFrame(self.metrics_report)

        csv_path = os.path.join(output_location, "Report.csv")

        self.metrics_report.to_csv(csv_path, index=False)
        print(f"mIoU report saved to {csv_path}")

