from feature_viewer.visergui import ViserViewer
from feature_viewer.renderer import renderer
import torch 
from gs_loader.feature_mapper import feature_lift_mapper, feature_lift_mapper_config
from gs_loader.kernel_loader import base_kernel_loader_config, general_gaussian_loader


import argparse

def parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--gs_ckpt', required=True, help="The place one put the gaussian ckpt, i.e. pretrained model")
    parser.add_argument('--feat_pt', required=True, help="place where one keep there features as pt")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parser()

    feature_lift_config = feature_lift_mapper_config(text_tokenizer='featup', text_encoder='maskclip')
    feature_mapper = feature_lift_mapper(config=feature_lift_config)

    gs_kernel_loader_config = base_kernel_loader_config(
        kernel_location=args.gs_ckpt,
        feature_location=args.feat_pt
    )

    gs_kernel_loader = general_gaussian_loader(gs_kernel_loader_config)


    with torch.no_grad():
        gs_renderer = renderer(
            gaussian_loader=gs_kernel_loader,
            feature_mapper=feature_mapper
        )

        viser = ViserViewer(
            device='cuda',
            viewer_port=6777
        )

        viser.set_renderer(gs_renderer)

        while True:
            viser.update()