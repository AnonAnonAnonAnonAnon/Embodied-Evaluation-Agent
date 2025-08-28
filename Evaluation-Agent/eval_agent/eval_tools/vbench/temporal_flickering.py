import numpy as np
from tqdm import tqdm
import cv2
from vbench.utils import load_dimension_info


def get_frames(video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


def mae_seq(frames):
    ssds = []
    for i in range(len(frames)-1):
        ssds.append(calculate_mae(frames[i], frames[i+1]))
    return np.array(ssds)


def calculate_mae(img1, img2):
    """Computing the mean absolute error (MAE) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


def cal_score(video_path):
    """please ensure the video is static"""
    frames = get_frames(video_path)
    score_seq = mae_seq(frames)
    return (255.0 - np.mean(score_seq).item())/255.0


def temporal_flickering(video_pairs):
    sim = []
    video_results = []
    
    for info in tqdm(video_pairs):
        query = info['prompt']
        video_path = info['content_path']
        
        try:
            score_per_video = cal_score(video_path)
        except AssertionError:
            continue
        video_results.append({'prompt':query, 'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)

    return {
        "score":[avg_score, video_results] 
    }


def compute_temporal_flickering(video_pairs):
    results = temporal_flickering(video_pairs)
    return results









