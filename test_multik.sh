python ./data_prepare_multipleK.py \
    --dataset_dir $DATASET_DIR \
    --model_dir $MODEL_DIR \
    --index_dir $INDEX_DIR \
    --context_type [vanilla_incontext, marking_incontext] \
    --save_dir $SAVE_DIR \
    --forbid_titles_dir ./wikitext103_forbidden_titles.txt \
    --splits [train, validation, test] \
    --construct_type [random, default] \
    --max_pos_embeddings 512 \
    --max_train_example 16000 \
    --retrieve_k $RETRIEVE_K
    
python test_multiple_k.py \
    --model_dir $MODEL_DIR \
    --adapter_dir $ADAPTER_DIR \
    --data_dir $DATA_DIR \
    --cache_dir $CACHE_DIR \
    --add_position back \
    --retrieve_k $RETRIEVE_K