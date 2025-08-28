import os
import sys
import argparse
import random
from omegaconf import OmegaConf
import torch
import torchvision

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts', 'evaluation')))

from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    load_image_batch,
    get_filelist,
)
from utils_vc.utils_vc import instantiate_from_config


class VideoCrafter:
    def __init__(self, mode="vc2"):
        """Load the model into memory to make running multiple predictions efficient"""
        self.mode = mode
        if self.mode == "vc2":
            ckpt_path_base = f"{CUR_DIR}/checkpoints/base_512_v2/model.ckpt"
            config_base = f"{CUR_DIR}/configs/inference_t2v_512_v2.0.yaml"
        else:
            raise Exception("Wrong Mode...")

        config_base = OmegaConf.load(config_base)
        model_config_base = config_base.pop("model", OmegaConf.create())
        self.model_base = instantiate_from_config(model_config_base)
        self.model_base = self.model_base.cuda()
        assert os.path.exists(ckpt_path_base), f"Error: checkpoint [{ckpt_path_base}] Not Found!"
        self.model_base = load_model_checkpoint(self.model_base, ckpt_path_base)
        self.model_base.eval()
    
    
    def predict(
        self, 
        prompt, 
        save_name, 
        ddim_steps: int = 50,
        seed: int = 123,
        save_fps: int = 10,
        unconditional_guidance_scale: float = 12.0
    ):
        
        width = 512
        height = 320
        model = self.model_base
        
        
        args = argparse.Namespace(
            mode="base",
            savefps=save_fps,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=1.0,
            bs=1,
            height=height,
            width=width,
            frames=-1,
            fps=28,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_guidance_scale_temporal=None,
        )
        
        ## sample shape
        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels

        batch_size = 1
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()
        prompts = [prompt]
        text_emb = model.get_learned_conditioning(prompts)


        cond = {"c_crossattn": [text_emb], "fps": fps}

        ## inference
        batch_samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
        )

        out_path = save_name # "/tmp/output.mp4"
        vid_tensor = batch_samples[0]
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            out_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )

