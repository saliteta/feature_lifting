import argparse
from evaluation.evaluator import lerf_evaluator
from gs_loader import kernel_loader, feature_mapper
from pathlib import Path

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', help="data colmap nerfstudio or scannet location of your nerfstudio initial dataset", type=str, required=True)
    parser.add_argument('--data_mode', help="data colmap nerfstudio or scannet location of your nerfstudio initial dataset", type=str, required=True)
    parser.add_argument('--feature_location', help="lifted feature location, feature.pt", type=str, required=True)
    parser.add_argument('--pretrained_location', help="the trained result like step-000029999.ckpt or ply file", type=str, required=True)
    parser.add_argument('--rendering_camera_location', help='camera_names for evaluation, should be a folder contain json and images', type=str, required=True)
    parser.add_argument('--rendered_result_location', help='rendered_result folder', type=str, required=True)
    parser.add_argument('--rendering_mode', help="select rendered result RGB, RGB+Feature, RGB+Feature+Feature_PCA", type=str, default="RGB+Feature+Feature_PCA")
    parser.add_argument('--saving_mode', help="saving feature mode", type=str, default="pt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()

    data_location = args.data_location
    data_mode = args.data_mode
    pretrained_location = args.pretrained_location
    feature_location:str = args.feature_location
    rendering_camera_location = args.rendering_camera_location
    rendered_result_location = args.rendered_result_location
    rendering_mode = args.rendering_mode
    saving_mode = args.saving_mode

    feature_lift_config = feature_mapper.feature_lift_mapper_config(text_tokenizer='featup', text_encoder='maskclip')
    feature_lift_mapper = feature_mapper.feature_lift_mapper(config=feature_lift_config)


    gs_kernel_loader_config = kernel_loader.base_kernel_loader_config(
        kernel_location=pretrained_location,
        feature_location=feature_location
    )

    gs_kernel_loader = kernel_loader.general_gaussian_loader(gs_kernel_loader_config)

    evaluator = lerf_evaluator(kernels_loader=gs_kernel_loader,
    feature_mapper = feature_lift_mapper,
    dataset_path=Path(data_location), 
    gt_paths=Path(rendering_camera_location),
    dataset_mode=data_mode)

    evaluator.load_camera()
    evaluator.eval(saving_path=Path(rendered_result_location), mode=rendering_mode,
    feature_saving_mode = saving_mode)

    