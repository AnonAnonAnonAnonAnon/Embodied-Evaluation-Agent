import os
import time
import argparse
import yaml, math
from tqdm import trange
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.distributed as dist

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


import sys
sys.path.append(CUR_DIR)

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from lvdm.utils.saving_utils import npz_to_video_grid, npz_to_imgsheet_5d
from scripts.sample_utils import load_model, get_conditions, make_model_input_shape, torch_to_np




def sample_denoising_batch(model, noise_shape, condition, *args,
                           sample_type="ddim", sampler=None, 
                           ddim_steps=None, eta=None,
                           unconditional_guidance_scale=1.0, uc=None,
                           denoising_progress=False,
                           **kwargs,
                           ):
    
    if sample_type == "ddpm":
        samples = model.p_sample_loop(cond=condition, shape=noise_shape,
                                      return_intermediates=False, 
                                      verbose=denoising_progress,
                                      **kwargs,
                                      )
    elif sample_type == "ddim":
        assert(sampler is not None)
        assert(ddim_steps is not None)
        assert(eta is not None)
        ddim_sampler = sampler
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                         conditioning=condition,
                                         batch_size=noise_shape[0],
                                         shape=noise_shape[1:],
                                         verbose=denoising_progress,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=uc,
                                         eta=eta,
                                         **kwargs,
                                        )
    else:
        raise ValueError
    return samples



@torch.no_grad()
def sample_text2video(model, prompt, n_samples, batch_size,
                      sample_type="ddim", sampler=None, 
                      ddim_steps=50, eta=1.0, cfg_scale=7.5, 
                      decode_frame_bs=1,
                      ddp=False, all_gather=True, 
                      batch_progress=True, show_denoising_progress=False,
                      num_frames=None,
                      ):
    # get cond vector
    assert(model.cond_stage_model is not None)
    cond_embd = get_conditions(prompt, model, batch_size)
    uncond_embd = get_conditions("", model, batch_size) if cfg_scale != 1.0 else None

    # sample batches
    all_videos = []
    n_iter = math.ceil(n_samples / batch_size)
    iterator  = trange(n_iter, desc="Sampling Batches (text-to-video)") if batch_progress else range(n_iter)
    for _ in iterator:
        noise_shape = make_model_input_shape(model, batch_size, T=num_frames)
        samples_latent = sample_denoising_batch(model, noise_shape, cond_embd,
                                            sample_type=sample_type,
                                            sampler=sampler,
                                            ddim_steps=ddim_steps,
                                            eta=eta,
                                            unconditional_guidance_scale=cfg_scale, 
                                            uc=uncond_embd,
                                            denoising_progress=show_denoising_progress,
                                            )
        samples = model.decode_first_stage(samples_latent, decode_bs=decode_frame_bs, return_cpu=False)
        
        # gather samples from multiple gpus
        if ddp and all_gather:
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    
    all_videos = np.concatenate(all_videos, axis=0)
    assert(all_videos.shape[0] >= n_samples)
    return all_videos



def save_results(videos, save_name="results", save_fps=8):

    for i in range(videos.shape[0]):
        npz_to_video_grid(videos[i:i+1,...], save_name, fps=save_fps)





class VideoCrafter09:
    def __init__(self):

        config = OmegaConf.load(f"{CUR_DIR}/checkpoints/base_t2v/model_config.yaml")
        self.model, _, _ = load_model(config, f"{CUR_DIR}/checkpoints/base_t2v/model.ckpt")
        self.ddim_sampler = DDIMSampler(self.model)

    
        
    
    def predict(self, prompt, save_name):
        samples = sample_text2video(self.model, prompt, 1, 1,
                        sample_type='ddim', sampler=self.ddim_sampler,
                        ddim_steps=50, eta=1.0, 
                        cfg_scale=15.0,
                        decode_frame_bs=1,
                        ddp=False, show_denoising_progress=True,
                        num_frames=16,
                        )

        save_results(samples, save_name, save_fps=8)



