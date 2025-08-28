import os
import json

import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, tag2text_transform, CACHE_DIR
from vbench.third_party.tag2Text.tag2text import tag2text_caption

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_caption(model, image_arrays):
    caption, tag_predict = model.generate(image_arrays, tag_input = None, return_tag_predict = True)
    return caption

def check_generate(key_info, predictions):
    cur_cnt = 0
    key = key_info['scene']
    for pred in predictions:
        q_flag = [q in pred for q in key.split(' ')]
        if len(q_flag) == sum(q_flag):
            cur_cnt +=1
    return cur_cnt

def scene(model, video_pairs, device):
    success_frame_count, frame_count = 0,0
    video_results = []
    transform = tag2text_transform(384)
    
    for info in tqdm(video_pairs):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        scene_info = info['auxiliary_info']
        video_path = info['content_path']
        query = info["prompt"]
        
            
        video_array = load_video(video_path, num_frames=16, return_tensor=False, width=384, height=384)
        video_tensor_list = []
        for i in video_array:
            video_tensor_list.append(transform(i).to(device).unsqueeze(0))
        video_tensor = torch.cat(video_tensor_list)
        cur_video_pred = get_caption(model, video_tensor)
        cur_success_frame_count = check_generate(scene_info, cur_video_pred)
        cur_success_frame_rate = cur_success_frame_count/len(cur_video_pred)
        success_frame_count += cur_success_frame_count
        frame_count += len(cur_video_pred)
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': cur_success_frame_rate})
            
            
    success_rate = success_frame_count / frame_count

    return {
        "score":[success_rate, video_results] 
    }
        


def compute_scene(video_pairs):
    device = torch.device("cuda")
    submodules_dict = {
        "pretrained": f'{CACHE_DIR}/caption_model/tag2text_swin_14m.pth',
        "image_size":384, 
        "vit":"swin_b"
    }

    model = tag2text_caption(**submodules_dict)
    model.eval()
    model = model.to(device)
    logger.info("Initialize caption model success")
    

    results = scene(model, video_pairs, device)
    return results
