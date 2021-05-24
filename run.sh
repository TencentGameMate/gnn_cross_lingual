#!/bin/bash

mode=$1

# ===============================================
#  PLEASE SPECIFY THE FOLLOWING ARGUMENTS
# ===============================================
DATA_ROOT="./data"
src_lang="en"       # e.g. en
tgt_lang="es"       # e.g. de
domain="slu"      # one of 'music', 'dvd', 'books', 'xglue', 'slu'
xfmr="xlmr"         # one of 'xlmr', 'xlm', 'bert'
output="$DATA_ROOT/output/"
translate_data="$DATA_ROOT/translations/$domain/"
pos_cache_path="$DATA_ROOT/pos_tags"
# ===============================================
#  PLEASE SPECIFY THE ARGUMENTS ABOVE
# ===============================================

data_cache_path="$output/data_cache/${src_lang}_${tgt_lang}_${domain}_${xfmr}"
mbert_feature_path="$output/data_cache/features/${xfmr}-${domain}/"
sim_edge_path="$data_cache_path/sim_edges_${xfmr}"

if [ "$domain" == "music" ] || [ "$domain" == "dvd" ] || [ "$domain" == "books" ]; then
    num_classes=2
elif [ "$domain" == "xglue" ]; then
    num_classes=10
elif [ "$domain" == "slu" ]; then
    num_classes=12
else
    echo "Unknown domain $domain"
    exit
fi

if [ ! -d "$pos_cache_path/$domain" ]; then
    mkdir -p $pos_cache_path/$domain
fi
if [ ! -d "$data_cache_path" ]; then
    mkdir -p $data_cache_path
fi
if [ ! -d "$mbert_feature_path" ]; then
    mkdir -p $mbert_feature_path
fi


if [ "$mode" == "train" ]; then
    python train.py --source_path $DATA_ROOT/$src_lang/$domain/ \
                --target_path $DATA_ROOT/$tgt_lang/$domain \
                --tagger_root_path $pos_cache_path/$domain/ \
                --trans_cache_path $translate_data \
                --doc_doc_edge_file $sim_edge_path \
                --log_path $output/logfile \
                --src_lan $src_lang \
                --tgt_lan $tgt_lang \
                --transformer_type $xfmr \
                --doc_doc_edge_stored \
                --num_layers 2 \
                --lr 2e-5 \
                --train_batch_size 256 \
                --valid_batch_size 16384 \
                --out_emb_size 768 \
                --hidden_size 512 \
                --encode_maxlen 128 \
                --dropout 0.5 \
                --log_every 5 \
                --eval_every 1 \
                --num_epochs 15 \
                --num_classes $num_classes \
                --word_node_cnt 10000 \
                --data_cache_path $data_cache_path \
                --mbert_feature_file $mbert_feature_path \
                --mbert_model_path $DATA_ROOT/output/transformers/${src_lang}-${domain}/${xfmr}_2ep_lr4e-5_len128/ \
                --output_path $output
elif [ "$mode" == "finetune" ]; then
    python eval_bert_ft.py --model_save_path $DATA_ROOT/output/transformers/${src_lang}-${domain}/${xfmr}_2ep_lr4e-5_len128/ \
                --transformer_type $xfmr \
                --train_path $DATA_ROOT/$src_lang/$domain \
                --test_path $DATA_ROOT/$tgt_lang/$domain \
                --epoch_num 2 \
                --lr 4e-5 \
                --num_classes $num_classes \
                --batch_size 32 \
                --max_length 128 \
                --src_lan $src_lang \
                --tgt_lan $tgt_lang \
                --trans_cache_path $translate_data
fi

