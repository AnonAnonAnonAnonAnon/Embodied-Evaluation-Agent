# 从open_ended_eval复制，基于open_ended_eval copy开发

import os
from datetime import datetime
import argparse

from base_agent import BaseAgent

#ADD
# EEA新导的包
import json
from pathlib import Path

# ADD
# 替换为EEA的prompt模板
# from system_prompts import sys_prompts
from system_prompts_EEA import sys_prompts


from tools import ToolCalling, save_json



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

    # ADD
    # EEA追加的3个参数
    parser.add_argument("--plan_only", action="store_true", help="only plan sub-aspects and exit")
    parser.add_argument("--k_subaspects", type=int, default=5, help="number of sub-aspects to propose")
    parser.add_argument("--mode", type=str, default="embodied", choices=["t2i","t2v","embodied"], help="planning mode")

    args = parser.parse_args()
    return args



class EvalAgent:
    def __init__(self, sample_model="sdxl-1", save_mode="img"):
        self.tools = ToolCalling(sample_model=sample_model, save_mode=save_mode)
        self.sample_model = sample_model
        self.user_query = ""
    
    
    def init_agent(self):
        # initialize agent
        self.prompt_agent = BaseAgent(system_prompt=sys_prompts["open-prompt-sys"], use_history=False, temp=0.7)
        self.task_agent = BaseAgent(system_prompt=sys_prompts["open-plan-sys"], temp=0.7)
        
        # ADD
        # EEA添加的
        self.planner_agent = BaseAgent(system_prompt=sys_prompts["embodied-subaspect-planner-sys"], use_history=False, temp=0.2)


        
    def format_results(self, results):
        formatted_text = "Observation:\n\n"
        for item in results:
            formatted_text += f"Prompt: {item['Prompt']}\n"
            for question, answer in zip(item["Questions"], item["Answers"]):
                formatted_text += f"Question: {question} -- Answer: {answer}\n"
            formatted_text += "\n"
        return formatted_text


    def observe(self, sub_question):
        sub_query = f"User-query: {self.user_query}\n\nSub-aspect: {sub_question['Sub-aspect']}\nThought: {sub_question['Thought']}"
        pq_infos = self.prompt_agent(sub_query, parse=True)
        
        for item in pq_infos["Step 2"]:
            img_path = self.tools.sample([item["Prompt"]], self.image_folder)[0]["content_path"]
            item["img_path"] = img_path
            answer_list = []
            for question in item["Questions"]:
                answer = self.tools.vlm_eval(img_path, question)
                answer_list.append(answer.replace("\n\n", " "))
            item["Answers"] = answer_list
        
        sub_question["eval_results"] = pq_infos["Step 2"]
        return self.format_results(pq_infos["Step 2"])

    
    def update_info(self):
        folder_name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "-" + self.user_query.replace(" ", "_")
        self.save_path = f"./open_domain_results/{self.sample_model}/{folder_name}"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.image_folder = os.path.join(self.save_path, "images")
        self.file_name = os.path.join(self.save_path, f"open_domain_exploration_results.json")

        # ADD
        # EEA新增，创建新文件
        self.plan_folder = os.path.join(self.save_path, "plan")
        os.makedirs(self.plan_folder, exist_ok=True)


    def explore(self, query, all_chat=[]):
        self.user_query = query
        self.update_info()
        self.init_agent()

        all_chat.append(query)
        n = 0
        while True:

            task_response = self.task_agent(query, parse=True)
            if task_response.get("Plan"):
                all_chat.append(task_response)
                query = "continue"
                continue
            if task_response.get("Summary"):
                print("Finished!")
                all_chat.append(task_response)
                break
                
            query = self.observe(task_response)
            all_chat.append(task_response)
            
            if n > 9:
                break
            n += 1
        
        all_chat.append(self.task_agent.messages)
        save_json(all_chat, self.file_name)

# ADD
# EEA模块测试，仅接受用户指令，输出子方面
    def plan_subaspects(self, k=5, mode="embodied", taxonomy=None):
        """
        只做“子方面规划”：给定 user_query → 返回 K 个子方面（JSON 列表）
        """
        taxo_text = ""
        if taxonomy is not None:
            taxo_text = "\nTAXONOMY:\n" + json.dumps(taxonomy, ensure_ascii=False)

        msg = (
            f"MODE: {mode}\n"
            f"K: {k}\n"
            f"USER_QUERY: {self.user_query}\n"
            f"{taxo_text}"
        )

        # 期望模型返回纯 JSON；BaseAgent(parse=True) 若能直接解析则更好
        try:
            result = self.planner_agent(msg, parse=True)
        except Exception:
            result = self.planner_agent(msg)

        # 统一成 Python 对象（兜底解析）
        if isinstance(result, str):
            s = result
            try:
                result = json.loads(s)
            except Exception:
                start, end = s.find("["), s.rfind("]")
                if start != -1 and end != -1 and end > start:
                    result = json.loads(s[start:end+1])
                else:
                    raise

        # 轻量清洗：去重 & 字段兜底
        cleaned, seen = [], set()
        for it in result:
            name = (it.get("Sub-aspect") or it.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cleaned.append({
                "Sub-aspect": name,
                "Thought": (it.get("Thought") or it.get("rationale") or "").strip(),
                "Probes": it.get("Probes") or it.get("probes") or [],
                "Priority": float(it.get("Priority", 0.5)),
                "EvalType": it.get("EvalType") or it.get("eval_type") or "rule",
                "EvidenceHint": it.get("EvidenceHint") or it.get("evidence_hint") or ""
            })
        return cleaned



def main():
    args = parse_args()
    user_query = args.user_query
    open_agent = EvalAgent(sample_model=args.model, save_mode="img")
    
    # ADD
    # 先只测试规划为子方面这个模块
    ###
    open_agent.user_query = user_query
    open_agent.update_info()
    open_agent.init_agent()
    if args.plan_only:
        subaspects = open_agent.plan_subaspects(k=args.k_subaspects, mode=args.mode)
        out_path = os.path.join(open_agent.plan_folder, "subaspects.json")
        save_json(subaspects, out_path)
        print(f"[PLAN ONLY] Saved sub-aspects to: {out_path}")
        return
    ###

    open_agent.explore(user_query)



if __name__ == "__main__":
    main()











