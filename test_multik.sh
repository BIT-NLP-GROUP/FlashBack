python ./data_prepare_multipleK.py \
    --dataset_dir [] \
    --model_dir [] \
    --index_dir [] \
    --context_type [vanilla_incontext, marking_incontext] \
    --save_dir [] \
    --forbid_titles_dir ./wikitext103_forbidden_titles.txt \
    --splits [train, validation, test] \
    --construct_type [random, default] \
    --max_pos_embeddings 512 \
    --max_train_example 16000 \
    --retrieve_k []
    
python test_multiple_k.py \
    --model_dir [] \
    --adapter_dir [] \
    --data_dir [] \
    --cache_dir [] \
    --add_position back \
    --retrieve_k []