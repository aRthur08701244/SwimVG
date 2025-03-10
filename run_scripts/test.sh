dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="swimvg_dinov2.yaml"
gpu=7
split_name="testB" # "val", "testA", "testB" 
# Evaluation on the specified of the specified dataset
filename=$dataset_name"_$(date +%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=$gpu \
python \
-u test.py \
--config config/$dataset_name/$config_name \
--opts TEST.test_split $split_name \

