import os
import json

import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, CACHE_DIR
from vbench.third_party.grit_model import DenseCaptioning

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dect_from_grit(model, image_arrays):
    pred = []
    if type(image_arrays) is not list:
        image_arrays = image_arrays.numpy()
    with torch.no_grad():
        for frame in image_arrays:
            try:
                pred.append(set(model.run_caption_tensor(frame)[0][0][2]))
            except:
                pred.append(set())
    return pred

def check_generate(key_info, predictions):
    cur_cnt = 0
    for pred in predictions:
        if key_info in pred:
            cur_cnt+=1
    return cur_cnt

def object_class(model, video_pairs, device):
    success_frame_count, frame_count = 0,0
    video_results = []
    for info in tqdm(video_pairs):
        
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        
        object_info = info['auxiliary_info']
        video_path = info['content_path']
        query = info["prompt"]
        
            
        video_tensor = load_video(video_path, num_frames=16)
        cur_video_pred = get_dect_from_grit(model, video_tensor.permute(0,2,3,1))
        cur_success_frame_count = check_generate(object_info, cur_video_pred)
        cur_success_frame_rate = cur_success_frame_count/len(cur_video_pred)
        success_frame_count += cur_success_frame_count
        frame_count += len(cur_video_pred)
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': cur_success_frame_rate})
            
            
    success_rate = success_frame_count / frame_count

    return {
        "score":[success_rate, video_results] 
    }
        

def compute_object_class(video_pairs):
    device = torch.device("cuda")
    
    dense_caption_model = DenseCaptioning(device)
    submodules_dict = {
        "model_weight": f'{CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth'
    }
    dense_caption_model.initialize_model_det(**submodules_dict)
    logger.info("Initialize detection model success")

    results = object_class(dense_caption_model, video_pairs, device)
    return results
