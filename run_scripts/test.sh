dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="swimvg_dinov2.yaml"
gpu=0 # 7
split_name="testB" # "val", "testA", "testB"
model_dir="exp/refcoco/L_V14_64_8_512 2025-10-20-12-53-58/best_model.pth"
# Evaluation on the specified of the specified dataset
filename=$dataset_name"_$(date +%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=$gpu \
python \
-u test.py \
--config config/$dataset_name/$config_name \
--opts TEST.test_split $split_name \
       TEST.model_dir "$model_dir" \
