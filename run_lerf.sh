#!/usr/bin/env bash

# We need to first specify the dataset location
dataset_path="/data/lerf_ovs/dataset"
eval_path="/data/lerf_ovs/label"
output_path="rendered_result"

# We need to prepare our feature data



#cd FeatUp
#for dir in "$dataset_path"/*; do
#  if [ -d "$dir" ]; then
#    echo "Processing directory: $dir"
#    python featup.py --image_folder "$dir/images" --output_feature_folder "$dir/features"
#    # Create 'colmap' folder if it doesn't exist (optional)
#    # mkdir -p "$dir/colmap"
#  fi
#done
##
### We first run gaussian_training to get all lerf models
#for dir in "$dataset_path"/*; do
#  if [ -d "$dir" ]; then
#    echo "Processing directory: $dir"
#    # Create 'colmap' folder if it doesn't exist (optional)
#    # mkdir -p "$dir/outputs/colmap"
#
#    ns-train splatfacto --data "$dir" \
#                        --output-dir "$dir/outputs" \
#                        --viewer.quit-on-train-completion True
#  fi
#done
#cd ..
## We finally run our feature lifting algorithm
#cd gsplat_ext/
##
#for dir in "$dataset_path"/*; do
#  if [ -d "$dir" ]; then
#    echo "Processing directory: $dir"
#    # Capture the file paths into variables
#    model_ckpt=$(find "$dir/outputs" -type f -name "step-000029999.ckpt")
#    folder=$(dirname "$(find "$dir/outputs" -type f -name "step-000029999.ckpt" | tail -n 1)")
#
#
#    python gaussian_inverse_splating.py \
#      --data_location "$dir" \
#      --data_mode nerfstudio \
#      --pretrained_location "$model_ckpt" \
#      --feature_location "$dir/features/" \
#      --output_feature "$dir/gs_feature.pt"
#  fi
#done

cd evaluation

for dir in "$dataset_path"/*; do
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    # Capture the file paths into variables
    model_ckpt=$(find "$dir/outputs" -type f -name "step-000029999.ckpt")
    folder=$(dirname "$(find "$dir/outputs" -type f -name "step-000029999.ckpt" | tail -n 1)")
    dir_basename=$(basename "$dir")
    python -W ignore evaluation.py \
      --data_location "$dir" \
      --data_mode "nerfstudio" \
      --pretrained_location "$model_ckpt" \
      --feature_location "$dir/gs_feature.pt" \
      --rendering_camera_location "$eval_path/$dir_basename" \
      --rendered_result_location "$output_path/$dir_basename"
    

  fi
done

cd ..