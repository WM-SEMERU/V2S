VAR3=/scratch1/v2s/v2s/tf-dataset/output/data/v2s_label_map.pbtxt
IMAGES=$1
OUTPUT=$2

python create_touch_tf_record.py --data_dir=${IMAGES} \
        --output_dir=${OUTPUT} \
        --label_map_path=${VAR3}

