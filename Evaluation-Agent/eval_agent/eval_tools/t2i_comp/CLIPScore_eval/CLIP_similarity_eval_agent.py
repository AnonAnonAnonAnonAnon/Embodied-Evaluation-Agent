# image and text similarity
# ref https://github.com/openai/CLIP
import os
import torch
import clip
from PIL import Image
import numpy as np


def clipscore(model, preprocess, image_pairs, device):
    total = []
    results = []

    for info in image_pairs:
        image_path = info["content_path"]
        prompt = info["prompt"]

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(prompt).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)


            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

             # Calculate the cosine similarity between the image and text features
            cosine_similarity = (image_features @ text_features.T).squeeze().item()

        similarity = cosine_similarity
        results.append({'prompt':prompt, 'image_path': image_path, 'image_results': similarity})
        total.append(similarity)
    
    avg_score = np.mean(total)
    return {
        "score":[avg_score, results] 
    }
    
    
def calculate_clip_score(image_pairs):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    eval_results = clipscore(model, preprocess, image_pairs, device)
    return eval_results








