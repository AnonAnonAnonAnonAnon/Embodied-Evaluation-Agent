MASTER_PORT=$((12000 + $RANDOM % 20000))
PROMPT="UCF101-1.txt" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="results/ucf101/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

srun -p aigc-video --gres=gpu:0 --cpus-per-task=8 -N1 --quotatype=reserved -w SH-IDC1-10-140-37-160 \
-N1 --job-name=eval_videocrafter \
python sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 20 \
    --batch_size 1 \
    --seed 2 \
    --show_denoising_progress
