import os
import json
from datetime import datetime
import argparse
import Levenshtein

from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Eval-Agent-T2I-CompBench', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--user_query",
        type=str,
        required=True,
        help="user query",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="sdxl-1",
        help="model",
    )

    args = parser.parse_args()
    return args




def most_similar_string(prompt, string_list):
    similarities = [Levenshtein.distance(prompt, item) for item in string_list]
    most_similar_idx = similarities.index(min(similarities))
    return string_list[most_similar_idx]


def check_and_fix_prompt(chosed_prompts, prompt_list):
    results_dict={}

    for key, item in chosed_prompts.items():
        item["Prompt"] = most_similar_string(item["Prompt"], prompt_list)

        results_dict[key] = item
        
    return results_dict


def format_dimension_as_string(df, dimension_name):
    row = df.loc[df['Dimension'] == dimension_name]
    if row.empty:
        return f"No data found for dimension: {dimension_name}"
    
    formatted_string = (
        f"{row['Dimension'].values[0]}: "
        f"Very High -> {row['Very High'].values[0]}, "
        f"High -> {row['High'].values[0]}, "
        f"Moderate -> {row['Moderate'].values[0]}, "
        f"Low -> {row['Low'].values[0]}, "
        f"Very Low -> {row['Very Low'].values[0]}"
    )
    
    return formatted_string


class EvalAgent:
    def __init__(self, sample_model="sdxl-1", save_mode="img", refer_file="t2i_comp_dimension_scores.tsv"):
        self.tools = ToolCalling(sample_model=sample_model, save_mode=save_mode)
        self.sample_model = sample_model
        self.user_query = ""
        self.tsv_file_path = refer_file
        
    
    
    def init_agent(self):
        self.prompt_agent = BaseAgent(system_prompt=sys_prompts["t2i-comp-prompt-sys"], use_history=False, temp=0.7)
        self.plan_agent = BaseAgent(system_prompt=sys_prompts["t2i-comp-plan-sys"], use_history=True, temp=0.7)



    def sample_and_eval(self, designed_prompts, save_path, tool_name):
        prompts = [item["Prompt"] for _, item in designed_prompts.items()]
        video_pairs = self.tools.sample(prompts, save_path)
        eval_results = self.tools.eval(tool_name, video_pairs)
        return eval_results


    def reference_prompt(self, search_dim):
        search_item = search_dim.replace("_binding", "")
        file_path = f"./eval_tools/t2i_comp/prompt_file/{search_item}_val.txt"
        
        with open(file_path, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        
        return lines



    def format_eval_result(self, results, reference_table):
        question = results["Sub-aspect"]
        tool_name = results["Tool"]
        average_score = results["eval_results"]["score"][0]
        video_results = results["eval_results"]["score"][1]
        
        
        output = f"Sub-aspect: {question}\n"
        output += f"The score categorization table for the numerical results evaluated by the '{tool_name}' is as follows:\n{reference_table}\n\n"
        output += f"Observation: The evaluation results using '{tool_name}' are summarized below.\n"
        output += f"Average Score: {average_score:.4f}\n"
        output += "Detailed Results:\n"

        for i, video in enumerate(video_results, 1):
            prompt = video["prompt"]
            score = video["image_results"]
            output += f"\t{i}. Prompt: {prompt}\n"
            output += f"\tScore: {score:.4f}\n"
        
        return output
    
    
    
    def update_info(self):
        folder_name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "-" + self.user_query.replace(" ", "_")
        self.save_path = f"./eval_t2i_comp_results/{self.sample_model}/{folder_name}"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.image_folder = os.path.join(self.save_path, "images")
        self.file_name = os.path.join(self.save_path, f"eval_results.json")



    def explore(self, query, all_chat=[]):
        
        self.user_query = query
        self.update_info()
        self.init_agent()
        df = pd.read_csv(self.tsv_file_path, sep='\t')
        
        plan_query = query
        all_chat.append(plan_query)
        
        n = 0
        while True:
            
            plans = self.plan_agent(plan_query, parse=True)
            if plans.get("Analysis"):
                all_chat.append(plans)
                print("Finish!")
                break
            
            tool_name = plans["Tool"].lower().strip().replace(" ", "_")
            reference_table = format_dimension_as_string(df, plans["Tool"])
            
            prompt_query = json.dumps(plans)
            prompt_list = self.reference_prompt(tool_name)
            prompt_query = f"Context:\n{prompt_query}\n\nPrompt list:\n{json.dumps(prompt_list)}"

            designed_prompts = self.prompt_agent(prompt_query, parse=True)
            designed_prompts = check_and_fix_prompt(designed_prompts, prompt_list)


            plans["eval_results"] = self.sample_and_eval(designed_prompts, self.image_folder, tool_name)
            plan_query = self.format_eval_result(plans, reference_table=reference_table)
            all_chat.append(plans)
            
            if n > 9:
                break
            n += 1


        all_chat.append(self.plan_agent.messages)
        save_json(all_chat, self.file_name)




def main():
    args = parse_args()
    user_query = args.user_query
    eval_agent = EvalAgent(sample_model=args.model, save_mode="img")
    eval_agent.explore(user_query)      


if __name__ == "__main__":
    main()




