from openai import OpenAI
import json
import os

class BaseAgent:
    def __init__(self, system_prompt="", use_history=True, temp=0, top_p=1):

        self.use_history = use_history

        # self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        # if not self.api_key:
        #     raise ValueError("Environment variable OPENAI_API_KEY is not set or empty")
        # self.model = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
        # if not self.model:
        #     raise ValueError("Environment variable OPENAI_MODEL is not set or empty")
        # self.base_url = None
        # third_party_url = (os.getenv("OPENAI_BASE_URL", "") or os.getenv("OPENAI_API_BASE", "")).strip().rstrip("/")
        # if third_party_url:
        #     self.base_url = third_party_url
        # if self.base_url:
        #     self.client = OpenAI(
        #         api_key=self.api_key,
        #         base_url=self.base_url
        #     )
        # else:
        #     self.client = OpenAI(
        #         api_key=self.api_key
        #     )

        # self.client = OpenAI()
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="sk-xDai8Jxb9bLlXroX6bFiS9MF96fj0tqAe8Zc9mPV0xjDK98S",
            base_url="https://api.chatanywhere.tech/v1"
            # base_url="https://api.chatanywhere.org/v1"
        )
        self.system = system_prompt
        self.model = "gpt-4o"

        self.temp = temp
        self.top_p = top_p
        self.messages = []
        
        self.system = system_prompt
        if self.system:
            self.messages.append({"role": "system", "content": system_prompt})
    
    
    def __call__(self, message, parse=False):
        """
        直接调用实例时，相当于发送用户消息并得到回复。
        """
        
        self.messages.append({"role": "user", "content": message})
        result = self.generate(message, parse)
        self.messages.append({"role": "assistant", "content": result})

        if parse:
            try:
                result = self.parse_json(result)
            except:
                raise Exception("Error content is list below:\n", result)
            
        return result
        
    def generate(self, message, json_format):
        """
        生成回复
        """
        if self.use_history:
            input_messages = self.messages
        else:
            input_messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": message}
            ]
            
        
        if json_format:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                response_format = { "type": "json_object" }
                )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                )
        return response.choices[0].message.content
    
    
    def parse_json(self, response):
        """
        把模型返回的 JSON 字符串转成 Python 对象
        """
        return json.loads(response)

    
    def add(self, message: dict):
        """
        add 并非严格必要，但它提供了一个直接操作对话历史的接口，方便在特殊场景下使用。
        """
        self.messages.append(message)


