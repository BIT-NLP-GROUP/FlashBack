python data_prepare.py \
    --dataset_dir /home/16tb_hdd/lrh/downloads/datasets/wikitext \
    --model_dir /home/16tb_hdd/lrh/downloads/models/gpt2 \
    --index_dir /home/16tb_hdd/lrh/downloads/pyserini_index/wikipedia-dpr-100w \
    --algorithm_type vanilla_incontext \
    --save_dir /home/16tb_hdd/lrh/downloads/datasets/prepared_dataset \
    --forbid_titles_dir /home/rhliu/projects/RAG-is-all-you-need/wikitext103_forbidden_titles.txt \
    --splits validation \
    --construct_type random
#forbid_titles文件比较小(2.6k)
