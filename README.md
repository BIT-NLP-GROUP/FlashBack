# FLASHBACK : Efficient Retrieval-Augmented Language Modeling for Long Context Inference
This repository is the official implementation of FlashBack.

## Requirements

- transformers=4.35.2
- datasets=2.14.6
- pyserini=0.22.0
- deepspeed=0.14.0

## Preliminary
### Index Building
Convert raw dataset to jsonlines format.

`bash data2jsonl.sh`
```
python data2jsonl.py \
--dataset_dir $DATASET_DIR \
--output_dir $OUTPUT_DIR \
--chunk_size 128 \
--tokenizer_dir $MODEL_DIR
```

Build index based on jsonlines files.

`bash json2index.sh`

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $OUTPUT_DIR \
  --index $INDEX_DIR \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storeRaw
```
### Data Preparation 

`bash rawdata2prepared.sh`
```
python data_prepare.py \
    --dataset_dir $DATASET_DIR \
    --model_dir $MODEL_DIR \
    --index_dir $INDEX_DIR \
    --save_dir $SAVE_DIR \
    --retrieve_stride 16 \
    --forbid_titles_dir $FORBID_TITLES_DIR \
    --train_split $TRAIN_SPLIT \
    --validation_split $VALIDATION_SPLIT \
    --max_pos_embeddings 512 \
    --recall_max_width 128
```

## Language Modeling Training/Evaluation
```
deepspeed --include localhost:2,3 --master_port 12138 \
    run.py \
    --model_name_or_path $MODEL_DIR \
    --dataset_name $DATASET_DIR \
    --cache_dir $CACHE_DIR \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --eval_steps 6000 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --logging_steps 5 \
    --evaluation_strategy steps \
    --save_steps 6000 \
    --overwrite_cache \
    --context_type $CONTEXT_TYPE \
    --tune_params $TUNE_PARAMS \
    --add_position $ADD_POSITION \
    --low_cpu_mem_usage \
    --max_steps 5000 \
    --retrieval_stride 16 \
    #--torch_dtype bfloat16
```
### Results

|Model|A.P.|LoRA|M.T.|Wikitext|Arxiv|Freelaw|Stackexchange|
|-----|----|----|----|--------|-----|-------|-------------|
|OPT-1.3B|N.| | |16.72|9.64|8.58|7.78|
|OPT-1.3B|P.| | |15.02|9.59|8.35|7.63|
|OPT-1.3B|P.|✔️| |10.23|8.33|7.47|6.87|
|OPT-1.3B|P.|✔️|✔️|**10.22**|**8.31**|**7.45**|**6.87**|
|OPT-1.3B|A.| | |81.57|51.94|53.73|42.99|
|OPT-1.3B|A.|✔️| |12.65|11.38|10.11|9.13|
|OPT-1.3B|A.|✔️|✔️|**10.49**|**8.72**|**7.85**|**7.17**|
|OPT-2.7B|N.| | |14.50|8.72|7.74|6.99|
|OPT-2.7B|P.| | |13.14|8.68|7.56|6.88|
|OPT-2.7B|P.|✔️| |9.23|7.69|6.80|6.23|
|OPT-2.7B|P.|✔️|✔️|**9.23**|**7.68**|**6.78**|**6.22**|
|OPT-2.7B|A.| | |76.41|50.68|51.78|42.48|
|OPT-2.7B|A.|✔️| |11.58|10.88|9.31|8.51|
|OPT-2.7B|A.|✔️|✔️|**9.52**|**8.08**|**7.19**|**6.53**|
|OPT-6.7B|N.| | |12.30|7.74|6.94|6.22|
|OPT-6.7B|P.| | |11.20|7.73|6.83|6.15|
|OPT-6.7B|P.|✔️| |8.23|6.98|6.19|5.58|
|OPT-6.7B|P.|✔️|✔️|**8.24**|**6.99**|**6.18**|**5.58**|
|OPT-6.7B|A.| | |68.31|46.53|48.33|40.25|
|OPT-6.7B|A.|✔️| |10.54|10.92|9.27|8.04|
|OPT-6.7B|A.|✔️|✔️|**8.59**|**7.43**|**6.64**|**5.94**|

## Run-time Improvement Evaluation

`bash test_speed.sh`
```
python test_speed.py \
    --model_dir $MODEL_DIR \
    --T $T \
    --test_mode [front, back] \
    --init_length $INIT_LENGTH \
    --lora [True, False]
```

## Ablation

### Retrieval Stride
Tune "--retrieval_stride".

### Multiple Passages

`bash test_multik.sh`
```
python data_prepare_multipleK.py \
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
```