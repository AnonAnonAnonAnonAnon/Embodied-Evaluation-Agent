import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from vbench.utils import load_video, load_dimension_info, CACHE_DIR

def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def technical_quality(model, video_pairs, device, **kwargs):
    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    video_results = []
    
    for info in tqdm(video_pairs):
        query = info['prompt']
        video_path = info['content_path']
        
        images = load_video(video_path)
        images = transform(images, preprocess_mode)
        acc_score_video = 0.
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
            
        video_result = acc_score_video/len(images)
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': video_result/100.})
        
        
    average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
    # average_score = average_score / 100.
    
    return {
        "score":[average_score, video_results] 
    }


def compute_imaging_quality(video_pairs):
    device = torch.device("cuda")
    model_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
    kwargs = {
        'imaging_quality_preprocessing_mode' : 'longer'
    }

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    
    results = technical_quality(model, video_pairs, device, **kwargs)
    return results
