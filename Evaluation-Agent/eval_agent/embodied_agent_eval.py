import os
from datetime import datetime
import argparse
from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json
from lightrag import query_lightrag
from typing import Mapping, Any, Dict
from pathlib import Path
import json
import argparse
import sys
import read_demo
import out


# 参数导入
def parse_args():
    parser = argparse.ArgumentParser(description='Eval-Agent-Embodied-Domain', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--user_query",
        type=str,
        required=True,
        help="user query",
    )


    args = parser.parse_args()
    return args

def combine_prompt_demo_rag(prompt: str,
                            self_py_demo: Mapping[str, Any],
                            rag_retrieve: Mapping[str, Any]) -> Dict[str, Any]:
    """
    将三者合并为一个 JSON 可序列化的字典：
    {
      "prompt": <item["Prompt"] 的字符串>,
      "demo":   <{'xx.py': 'content', ...}>,
      "rag_retrieve": <rag_retrieve 原样字典>
    }

    参数
    ----
    prompt : str
        即 item["Prompt"]，字符串。
    self_py_demo : Mapping[str, Any]
        形如 {'demo': {'xx.py': 'content', ...}} 或直接 {'xx.py': 'content', ...}。
        函数会自动兼容两种输入形式。
    rag_retrieve : Mapping[str, Any]
        rag_retrieve 的字典（例如包含 'response' 键）。

    返回
    ----
    Dict[str, Any]
        可直接 json.dumps 的合并结果。
    """
    # 兼容 {'demo': {...}} 或直接 {...}
    if isinstance(self_py_demo, Mapping) and "demo" in self_py_demo and isinstance(self_py_demo["demo"], Mapping):
        demo_mapping = dict(self_py_demo["demo"])
    else:
        demo_mapping = dict(self_py_demo)

    return {
        "prompt": str(prompt),
        "demo": demo_mapping,
        "rag_retrieve": dict(rag_retrieve),
    }

def payload_to_json_str(prompt: str,
                        demo_mapping: Mapping[str, Any],
                        rag_retrieve: Mapping[str, Any],
                        pretty: bool = False) -> str:
    """
    将 {"prompt": prompt, "demo": demo_mapping, "rag_retrieve": rag_retrieve}
    序列化为 JSON 字符串。默认紧凑输出；pretty=True 时美化缩进。

    - 兼容 demo_mapping 既可为 {'demo': {...}} 也可为 {...}
    - ensure_ascii=False 保留中文
    - default=str 兜底处理不可序列化对象
    """
    # 兼容 {'demo': {...}} 或直接 {...}
    if isinstance(demo_mapping, Mapping) and "demo" in demo_mapping and isinstance(demo_mapping["demo"], Mapping):
        demo = dict(demo_mapping["demo"])
    else:
        demo = dict(demo_mapping)

    payload: Dict[str, Any] = {
        "prompt": str(prompt),
        "demo": demo,
        "rag_retrieve": dict(rag_retrieve),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        default=str
    )
class EvalAgent:
    def __init__(self):
        # tools对象 (图像生成/评估工具集)
        # self.tools = ToolCalling()
        # 初始化
        self.user_query = ""
    
    
    def init_agent(self):
        # 初始化两个子代理，分别用于子问题生成和任务规划。
        self.prompt_agent = BaseAgent(system_prompt=sys_prompts["em-prompt-sys"], use_history=False, temp=0.7)
        self.task_agent = BaseAgent(system_prompt=sys_prompts["em-plan-sys"], temp=0.7)
        self.code_agent = BaseAgent(system_prompt=sys_prompts["em-code-sys"], temp=0.7,use_history=True)
        self.py_demo = read_demo.read()
        
        
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
            # 生成代码
            # img_path = self.tools.sample([item["Prompt"]], self.image_folder)[0]["content_path"]
            self.rag_retrieve = query_lightrag(item["Prompt"])
            input=payload_to_json_str(item["Prompt"],self.py_demo,self.rag_retrieve)

            result=self.code_agent(input,parse=True)
            print(result)
            paths = out.dump_py_files(result, out_dir="./output")
            print("Written:", *map(str, paths), sep="\n- ")
            #到此为止，正式生成代码
            #
            # item["img_path"] = img_path
            # answer_list = []
            # for question in item["Questions"]:
            #     answer = self.tools.vlm_eval(img_path, question)
            #     # 输入: (图像路径, question)
            #     # 输出: answer (VLM答案)
            #     answer_list.append(answer.replace("\n\n", " "))
            # item["Answers"] = answer_list
        
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
        # self.update_info()
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
    open_agent = EvalAgent()
    # 开始探索
    open_agent.explore(user_query)

if __name__ == "__main__":
    main()











