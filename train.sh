python data_prepare.py \
     --dataset_dir [] \
    --model_dir [] \
    --index_dir [] \
    --context_type [vanilla_incontext, marking_incontext] \
    --save_dir [] \
    --forbid_titles_dir ./wikitext103_forbidden_titles.txt \
    --splits [train, validation, test] \
    --construct_type [random, default] \
    --max_pos_embeddings 512 \
    --max_train_example 16000

deepspeed --include localhost:0 --master_port 12138 \
    run.py \
    --model_name_or_path [] \
    --dataset_name [] \
    --cache_dir [] \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --output_dir [] \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --eval_steps 200 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --logging_steps 5 \
    --evaluation_strategy steps \
    --save_steps 1000 \
    --overwrite_cache \
    --context_type [vanilla_incontext, marking_incontext] \
    --tune_params [all_params, added_params] \
    --add_position [front, back] \
    --low_cpu_mem_usage \
    --max_steps 1000 
    # --torch_dtype bfloat16 \
