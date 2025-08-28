from diffusers import DiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusion3Pipeline


class SD14:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.to("cuda")

    def predict(self, prompt, save_name):
        image = self.pipe(prompt=prompt).images[0]
        image.save(save_name)


class SD21:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-1"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

    def predict(self, prompt, save_name):
        image = self.pipe(prompt=prompt).images[0]
        image.save(save_name)



class SDXL:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")

    def predict(self, prompt, save_name):
        image = self.pipe(prompt=prompt).images[0]
        image.save(save_name)


class SD3:
    def __init__(self, model_name="stabilityai/stable-diffusion-3-medium-diffusers"):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
    
    def predict(self, prompt, save_name):
        image = self.pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        image.save(save_name)
