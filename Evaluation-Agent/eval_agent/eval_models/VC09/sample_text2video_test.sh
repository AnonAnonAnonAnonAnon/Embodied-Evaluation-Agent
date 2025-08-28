
PROMPT="a cat wearing sunglasses and working as a lifeguard at pool." # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="/mnt/petrelfs/zhangfan.p/zhangfan/evaluate-agent/agent/models/VC_09/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

srun -p video-aigc-3 --gres=gpu:1 --quotatype=reserved --cpus-per-task=16 --job-name=eval python sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 5 \
    --batch_size 1 \
    --seed 2 \
    --show_denoising_progress
