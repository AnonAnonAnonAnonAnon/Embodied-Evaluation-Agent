from openai import OpenAI
import json


class BaseAgent:
    def __init__(self, system_prompt="", use_history=True, temp=0, top_p=1):
        self.use_history = use_history
        # self.client = OpenAI()
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="sk-xDai8Jxb9bLlXroX6bFiS9MF96fj0tqAe8Zc9mPV0xjDK98S",
            base_url="https://api.chatanywhere.tech/v1"
            # base_url="https://api.chatanywhere.org/v1"
        )
        self.system = system_prompt
        self.temp = temp
        self.top_p = top_p
        self.input_tokens_count = 0
        self.output_tokens_count = 0
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system_prompt})
    
    
    def __call__(self, message, parse=False):
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
        if self.use_history:
            input_messages = self.messages
        else:
            input_messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": message}
            ]
            
        
        if json_format:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06", # gpt-4
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                response_format = { "type": "json_object" }
                )
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06", # gpt-4
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                )
        self.update_tokens_count(response)
        return response.choices[0].message.content
    
    
    def parse_json(self, response):
        return json.loads(response)

    
    def add(self, message: dict):
        self.messages.append(message)
    
    
    def update_tokens_count(self, response):
        self.input_tokens_count += response.usage.prompt_tokens
        self.output_tokens_count += response.usage.completion_tokens
    
    
    def show_usage(self):
        print(f"Total input tokens used: {self.input_tokens_count}\nTotal output tokens used: {self.output_tokens_count}")
        



