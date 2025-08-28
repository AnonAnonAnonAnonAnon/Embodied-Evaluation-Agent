# Please update the version of diffusers at leaset to 0.30.0
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from torchvision.utils import save_image
import torch
import torchvision
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class Latte1:
    def __init__(self):
        self.model_path = f"{CUR_DIR}/checkpoints/Latte-1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.video_length = 16 # 1 (text-to-image) or 16 (text-to-video)
        self.pipe = LattePipeline.from_pretrained(self.model_path, torch_dtype=torch.float16).to(self.device) # "maxin-cn/Latte-1"
        # Using temporal decoder of VAE
        vae = AutoencoderKLTemporalDecoder.from_pretrained(self.model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(self.device) # "maxin-cn/Latte-1"
        self.pipe.vae = vae
    
    def predict(self, prompt, save_name):
        videos = self.pipe(prompt, video_length=self.video_length, output_type='pt').frames.cpu()
        videos = (videos.clamp(0, 1) * 255).to(dtype=torch.uint8)
        video_ = videos[0].permute(0, 2, 3, 1)
        torchvision.io.write_video(save_name, video_, fps=8)

