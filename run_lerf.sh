#!/usr/bin/env bash

# Prompt user for dataset path
read -p "Enter dataset path [default: ./datasets/lerf_ovs/dataset]: " dataset_path
dataset_path=${dataset_path:-./datasets/lerf_ovs/dataset}

# Prompt user for evaluation path
read -p "Enter eval path [default: ./datasets/lerf_ovs/label]: " eval_path
eval_path=${eval_path:-./datasets/lerf_ovs/label}

# Prompt user for output path
read -p "Enter output path [default: ./evaluation/rendered_result]: " output_path
output_path=${output_path:-./evaluation/rendered_result}

# Print results
echo "Using dataset path: $dataset_path"
echo "Using eval path: $eval_path"
echo "Using output path: $output_path"

# We need to prepare our feature data



#for dir in "$dataset_path"/*; do
#  if [ -d "$dir" ]; then
#    echo "Processing directory: $dir"
#    python featup_encoder.py --image_folder "$dir/images" --output_feature_folder "$dir/features"
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
