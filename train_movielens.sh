# 1小时等于3600秒
SLEEP_SECONDS=3600

# 睡眠1小时
sleep $SLEEP_SECONDS

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--mode train \
--batch_size 8 \
--accumulate_grad_batches 8 \
--dataset movielens_data \
--data_dir /mnt/bn/data-tns-live-llm/leon/LLaRA-similar_seq_as_demo-/data/LLaRA/movielens \
--cans_num 20 \
--prompt_path /mnt/bn/data-tns-live-llm/leon/LLaRA-similar_seq_as_demo-/prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model \
--rec_model_path ./rec_model/movielens.pt \
--output_dir /mnt/bn/data-tns-live-llm/leon/datasets/rec/movielens \
--log_dir movielens_logs \
--lr_warmup_start_lr 2e-6 \
--lr 2e-4 \
--lr_decay_min_lr 2e-6 \
--max_epochs 5 \
--precision bf16 \
--unsloth 0

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--mode train \
--batch_size 8 \
--accumulate_grad_batches 8 \
--dataset movielens_data \
--data_dir /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/data/ref/movielens \
--cans_num 20 \
--prompt_path /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/  \
--rec_model_path ./rec_model/movielens.pt \
--output_dir /mnt/bn/data-tns-live-llm/leon/datasets/rec/movielens_unsloth_3 \
--log_dir movielens_logs \
--lr_warmup_start_lr 2e-6 \
--lr 2e-4 \
--lr_decay_min_lr 2e-6 \
--max_epochs 4 \
--precision bf16 \
--unsloth 1 \
--adapter_path /mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model_adapter_3 \
--score_model_path /mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model_3

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--mode train \
--batch_size 8 \
--accumulate_grad_batches 8 \
--dataset movielens_data \
--data_dir /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/data/ref/movielens \
--cans_num 20 \
--prompt_path /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/  \
--rec_model_path ./rec_model/movielens.pt \
--output_dir /mnt/bn/data-tns-live-llm/leon/datasets/rec/movielens_unsloth_5 \
--log_dir movielens_logs \
--lr_warmup_start_lr 2e-6 \
--lr 2e-4 \
--lr_decay_min_lr 2e-6 \
--max_epochs 4 \
--precision bf16 \
--unsloth 1 \
--adapter_path /mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model_adapter_5 \
--score_model_path /mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model_5

# # game
# CUDA_VISIBLE_DEVICES=6 python3 main.py \
# --mode train \
# --batch_size 8 \
# --accumulate_grad_batches 8 \
# --dataset steam_data \
# --data_dir /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/data/ref/game \
# --cans_num 20 \
# --prompt_path /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-Game/prompt/game.txt \
# --rec_embed SASRec \
# --llm_tuning lora \
# --llm_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-2-7b-bnb-4bit  \
# --rec_model_path ./rec_model/movielens.pt \
# --output_dir /mnt/bn/data-tns-live-llm/leon/datasets/rec/game_unsloth \
# --log_dir game_logs \
# --lr_warmup_start_lr 2e-6 \
# --lr 2e-4 \
# --lr_decay_min_lr 2e-6 \
# --max_epochs 5 \
# --precision bf16 \
# --unsloth 1 

# # lastfm
# CUDA_VISIBLE_DEVICES=5 python3 main.py \
# --mode train \
# --batch_size 8 \
# --accumulate_grad_batches 8 \
# --dataset lastfm_data \
# --data_dir /mnt/bn/data-tns-live-llm/leon/recom/LLaRA-similar_seq_as_demo-/data/ref/lastfm \
# --cans_num 20 \
# --prompt_path /mnt/bn/data-tns-live-llm/leon/recom/LLaRA_Lastfm/prompt/artist.txt \
# --rec_embed SASRec \
# --llm_tuning lora \
# --llm_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-2-7b-bnb-4bit  \
# --rec_model_path ./rec_model/movielens.pt \
# --output_dir /mnt/bn/data-tns-live-llm/leon/datasets/rec/lastfm_unsloth \
# --log_dir lastfm_logs \
# --lr_warmup_start_lr 2e-6 \
# --lr 2e-4 \
# --lr_decay_min_lr 2e-6 \
# --max_epochs 5 \
# --precision bf16 \
# --unsloth 1 