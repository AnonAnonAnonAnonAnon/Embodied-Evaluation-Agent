MASTER_PORT=$((12000 + $RANDOM % 20000))
PROMPT="/mnt/petrelfs/yujiashuo/Large-Video-Inference/txt/VideoAIGC-Eval-PromptPreparation-2023-08/prompt_by_class/pexels_selected/transportation_selected100_en.txt" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="/mnt/petrelfs/share_data/yujiashuo/sample/sample_videocraft/transportation_selected100_en/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

srun -p Video-aigc-general --gres=gpu:1 --cpus-per-task=16 -N1 --quotatype=reserved --async \
-N1 --job-name=eval_videocrafter \
python sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 5 \
    --batch_size 1 \
    --seed 2 \
    --show_denoising_progress