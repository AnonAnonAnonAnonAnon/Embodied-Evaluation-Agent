import os
import json
import logging
import numpy as np
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from vbench.utils import load_video, load_dimension_info, clip_transform, CACHE_DIR
from tqdm import tqdm


def background_consistency(clip_model, preprocess, video_pairs, device):
    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform(224)
    for info in tqdm(video_pairs):
        video_sim = 0.0
        
        query = info['prompt']
        video_path = info['content_path']

        images = load_video(video_path)
        images = image_transform(images)
            
    
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)
        sim += video_sim
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': sim_per_image})

    sim_per_frame = sim / cnt

    return {
        "score":[sim_per_frame, video_results] 
    }


def compute_background_consistency(video_pairs):
    device = torch.device("cuda")
    vit_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
    clip_model, preprocess = clip.load(vit_path, device=device)

    results = background_consistency(clip_model, preprocess, video_pairs, device)
    return results

