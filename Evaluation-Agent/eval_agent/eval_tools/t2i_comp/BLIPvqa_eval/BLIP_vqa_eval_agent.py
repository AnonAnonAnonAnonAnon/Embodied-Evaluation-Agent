import argparse
import os

import torch

from tqdm import tqdm, trange


import json
from tqdm.auto import tqdm
import sys
import spacy

from eval_tools.t2i_comp.BLIPvqa_eval.BLIP.train_vqa_func import VQA_main
import shutil
import secrets
import string

def Create_annotation_for_BLIP(image_pairs, outpath, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    cnt=0

    for info in image_pairs:
        
        image_dict={}
        image_dict['image'] = info["content_path"]
        image_dict['question_id']= cnt
        f = info["prompt"]
        doc = nlp(f)
        
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ['top', 'the side', 'the left', 'the right']:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if(len(noun_phrases)>np_index):
            q_tmp = noun_phrases[np_index]
            image_dict['question']=f'{q_tmp}?'
        else:
            image_dict['question'] = ''
            

        image_dict['dataset']="color"
        cnt+=1

        annotations.append(image_dict)

    print('Number of Processed Images:', len(annotations))

    json_file = json.dumps(annotations)
    with open(f'{outpath}/vqa_test.json', 'w') as f:
        f.write(json_file)



def blip_vqa(out_dir, image_pairs, np_num=8):

    np_index = np_num #how many noun phrases

    answer = []
    sample_num = len(image_pairs)
    reward = torch.zeros((sample_num, np_index)).to(device='cuda')


    order="_blip" #rename file
    for i in tqdm(range(np_index)):
        print(f"start VQA{i+1}/{np_index}!")
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            image_pairs,
            f"{out_dir}/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}{order}/",
                              f"{out_dir}/annotation{i + 1}{order}/VQA/")
        answer.append(answer_tmp)

        with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if(r_tmp[k]['question']!=''):
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i+1}/{np_index}!")
    reward_final = reward[:,0]
    for i in range(1,np_index):
        reward_final *= reward[:,i]

    #output final json
    with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
        r = json.load(file)
    reward_after=0
    for k in range(len(r)):
        r[k]["answer"] = '{:.4f}'.format(reward_final[k].item())
        reward_after+=float(r[k]["answer"])
    
    results = []
    for info_image, info_result in zip(image_pairs, r): 
        results.append({'prompt':info_image["prompt"], 'image_path': info_image["content_path"], 'image_results': float(info_result["answer"])})
    
    avg_score = reward_after/len(r)
    
    return {
        "score":[avg_score, results] 
    }



def generate_secure_random_number(length):
    digits = string.digits
    secure_random_number = ''.join(secrets.choice(digits) for i in range(length))
    return secure_random_number


def calculate_attribute_binding(image_pairs):
    random_string = image_pairs[-1]["prompt"].replace(" ","_")+generate_secure_random_number(7)
    out_path = "./folder_temporary_" + random_string
    eval_results = blip_vqa(out_path, image_pairs, np_num=8)
    shutil.rmtree(out_path)
    return eval_results


