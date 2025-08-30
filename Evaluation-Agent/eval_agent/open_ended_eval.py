import os
from datetime import datetime
import argparse

from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json

# 参数导入
def parse_args():
    parser = argparse.ArgumentParser(description='Eval-Agent-Open-Domain', formatter_class=argparse.RawTextHelpFormatter)

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
        help="model",
    )

    args = parser.parse_args()
    return args

class EvalAgent:
    def __init__(self, sample_model="sdxl-1", save_mode="img"):
        # tools对象 (图像生成/评估工具集)
        self.tools = ToolCalling(sample_model=sample_model, save_mode=save_mode)
        # 初始化
        self.sample_model = sample_model
        self.user_query = ""
    
    
    def init_agent(self):
        # 初始化两个子代理，分别用于子问题生成和任务规划。
        self.prompt_agent = BaseAgent(system_prompt=sys_prompts["open-prompt-sys"], use_history=False, temp=0.7)
        self.task_agent = BaseAgent(system_prompt=sys_prompts["open-plan-sys"], temp=0.7)
        
        
    def format_results(self, results):
        # 输出: 整个评估结果文本 (str)
        formatted_text = "Observation:\n\n"
        for item in results:
            formatted_text += f"Prompt: {item['Prompt']}\n"
            for question, answer in zip(item["Questions"], item["Answers"]):
                formatted_text += f"Question: {question} -- Answer: {answer}\n"
            formatted_text += "\n"
        return formatted_text


    def observe(self, sub_question):
        # 格式化的子问题字符串
        sub_query = f"User-query: {self.user_query}\n\nSub-aspect: {sub_question['Sub-aspect']}\nThought: {sub_question['Thought']}"
        # 子问题字符串
        pq_infos = self.prompt_agent(sub_query, parse=True)
        
        for item in pq_infos["Step 2"]:
            # 生成的图像路径 img_path
            img_path = self.tools.sample([item["Prompt"]], self.image_folder)[0]["content_path"]
            item["img_path"] = img_path
            answer_list = []
            for question in item["Questions"]:
                answer = self.tools.vlm_eval(img_path, question)
                # 输入: (图像路径, question)
                # 输出: answer (VLM答案)
                answer_list.append(answer.replace("\n\n", " "))
            item["Answers"] = answer_list
        
        sub_question["eval_results"] = pq_infos["Step 2"]
        # 在 sub_question 中写入结果
        return self.format_results(pq_infos["Step 2"])

    
    def update_info(self):
        # 创建保存路径，准备结果文件。
        folder_name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "-" + self.user_query.replace(" ", "_")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.save_path = f"{BASE_DIR}/open_domain_results/{self.sample_model}/{folder_name}"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.image_folder = os.path.join(self.save_path, "images")
        self.file_name = os.path.join(self.save_path, f"open_domain_exploration_results.json")


    def explore(self, query, all_chat=[]):
        self.user_query = query
        self.update_info()
        self.init_agent()

        # 聊天历史添加用户问题
        all_chat.append(query)
        n = 0
        while True:

            task_response = self.task_agent(query, parse=True)
            # 如果有 Plan, 继续迭代
            if task_response.get("Plan"):
                all_chat.append(task_response)
                query = "continue"
                continue
            # 如果有 Summary, 停止
            if task_response.get("Summary"):
                print("Finished!")
                all_chat.append(task_response)
                break
                
            query = self.observe(task_response)
            all_chat.append(task_response)
            
            # 最多 10 次循环
            if n > 9:
                break
            n += 1
        
        all_chat.append(self.task_agent.messages)
        save_json(all_chat, self.file_name)



def main():
    args = parse_args()
    user_query = args.user_query
    # 初始化评估代理
    open_agent = EvalAgent(sample_model=args.model, save_mode="img")
    # 开始探索
    open_agent.explore(user_query)

if __name__ == "__main__":
    main()











