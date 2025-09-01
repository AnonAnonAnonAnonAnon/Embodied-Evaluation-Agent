# Please update the version of diffusers at leaset to 0.30.0
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from torchvision.utils import save_image
import torch
import torchvision


class Latte1:
    def __init__(self):
        # self.model_path = f"{CUR_DIR}/checkpoints/Latte-1"
        self.model_name = "maxin-cn/Latte-1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.video_length = 16 # 1 (text-to-image) or 16 (text-to-video)
        self.pipe = LattePipeline.from_pretrained(self.model_name, torch_dtype=torch.float16, resume_download=True).to(self.device) # "maxin-cn/Latte-1"
        
                # ===== 省显存开关（视频管线同样适用）=====
        self.pipe.enable_attention_slicing()   # 注意力切片，减少峰值激活显存
        self.pipe.enable_vae_slicing()         # VAE 切片
        self.pipe.enable_vae_tiling()          # VAE 平铺（分块解码视频帧，更省显存）
        try:
            self.pipe.enable_xformers_memory_efficient_attention()  # 如可用，进一步省显存
        except Exception:
            pass
        
        # Using temporal decoder of VAE
        vae = AutoencoderKLTemporalDecoder.from_pretrained(self.model_name, subfolder="vae_temporal_decoder", torch_dtype=torch.float16, resume_download=True).to(self.device) # "maxin-cn/Latte-1"
        self.pipe.vae = vae
    
    def predict(self, prompt, save_name):
        videos = self.pipe(
            prompt, video_length=self.video_length, num_inference_steps=16, output_type='pt'
        ).frames.cpu()
        videos = (videos.clamp(0, 1) * 255).to(dtype=torch.uint8)
        video_ = videos[0].permute(0, 2, 3, 1)
        torchvision.io.write_video(save_name, video_, fps=8)

