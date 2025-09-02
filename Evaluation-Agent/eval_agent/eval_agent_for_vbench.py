import re, time, os
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import Levenshtein

from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description='Eval-Agent-VBench', formatter_class=argparse.RawTextHelpFormatter)

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
        default="latte1",
        help="target model",
    )

    args = parser.parse_args()
    return args

# 工具：在候选prompt列表中找与目标最相似的prompt（编辑距离最小）
def most_similar_string(prompt, string_list):
    similarities = [Levenshtein.distance(prompt, item["Prompt"]) for item in string_list]
    most_similar_idx = similarities.index(min(similarities))
    return string_list[most_similar_idx]

# 工具：用参考库prompt替换设计prompt（保持schema一致），但保留Thought
# chosed_prompts = {
#     "Step 1": {"Prompt": "A red cat", "Thought": "test cat motion"}
# }
# prompt_list = [
#     {"Prompt": "A red cat jumps"},
#     {"Prompt": "A dog running"}
# ]
# ->
# {
#   "Step 1": {
#     "Prompt": "A red cat jumps",     # 来自参考库，schema 正确
#     "Thought": "test cat motion"     # 保留原有思考内容
#   }
# }
def check_and_fix_prompt(chosed_prompts, prompt_list):
    results_dict={}

    for key, item in chosed_prompts.items():
        thought = item["Thought"]
        sim_item = most_similar_string(item["Prompt"], prompt_list)
        sim_item["Thought"] = thought
        results_dict[key] = sim_item
        
    return results_dict

# 将VBench评分表某一维度格式化为字符串（用于结果说明）
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
    def __init__(self, sample_model="latte1", save_mode="video", refer_file="vbench_dimension_scores.tsv"):
        self.tools = ToolCalling(sample_model=sample_model, save_mode=save_mode)
        self.sample_model = sample_model
        self.user_query = ""
        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        self.tsv_file_path = os.path.join(CUR_DIR,refer_file)

    
    def init_agent(self):

        self.prompt_agent = BaseAgent(system_prompt=sys_prompts["vbench-prompt-sys"], use_history=False, temp=0.7)
        self.plan_agent = BaseAgent(system_prompt=sys_prompts["vbench-plan-sys"], use_history=True, temp=0.7)


    # 找到和指定 prompt 匹配的 辅助信息 auxiliary_info
    def search_auxiliary(self, designed_prompts, prompt):
        for _, value in designed_prompts.items():
            if value['Prompt'] == prompt:
                return value["auxiliary_info"]
        raise ValueError("Didn't find auxiliary info, please check your json."
)

    def sample_and_eval(self, designed_prompts, save_path, tool_name):
        # 提取 prompt → 生成视频（self.tools.sample）
        # 绑定辅助信息
        # 调用评测工具对视频进行打分
        prompts = [item["Prompt"] for _, item in designed_prompts.items()]
        video_pairs = self.tools.sample(prompts, save_path)
        if 'auxiliary_info' in designed_prompts["Step 1"]:
            for item in video_pairs:
                item["auxiliary_info"] = self.search_auxiliary(designed_prompts, item["prompt"])
        
        eval_results = self.tools.eval(tool_name, video_pairs)
        return eval_results


    def reference_prompt(self, search_dim):
        '''
        1) 功能定位：
        - 从 VBench 参考库中检索“包含目标维度(search_dim)”的条目；
        - 统一字段格式（prompt_en → Prompt，移除 dimension）；
        - 提取该维度的 auxiliary_info（如 hint），便于后续生成/评测环节使用。
        - 归属“Prompt参考检索 / 标准化映射”功能，不属于参数导入。
        '''
        file_path = "./eval_tools/vbench/VBench_full_info.json"
        data = json.load(open(file_path, "r"))

        results = []
        for item in data:
            if search_dim in item["dimension"]:
                item.pop("dimension")
                item["Prompt"] = item.pop("prompt_en")
                if 'auxiliary_info' in item and search_dim in item['auxiliary_info']:
                    item["auxiliary_info"] = list(item["auxiliary_info"][search_dim].values())[0]
                results.append(item)
        
        return results

    # 把数值结果拼成一段可读的总结文本
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
            score = video["video_results"]
            output += f"\t{i}. Prompt: {prompt}\n"
            output += f"\tScore: {score:.4f}\n"
        
        return output


    def update_info(self):
        folder_name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "-" + self.user_query.replace(" ", "_")
        self.save_path = f"./eval_vbench_results/{self.sample_model}/{folder_name}"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.video_folder = os.path.join(self.save_path, "videos")
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

            plans["eval_results"] = self.sample_and_eval(designed_prompts, self.video_folder, tool_name)
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
    eval_agent = EvalAgent(sample_model=args.model, save_mode="video")
    eval_agent.explore(user_query)


if __name__ == "__main__":
    main()
