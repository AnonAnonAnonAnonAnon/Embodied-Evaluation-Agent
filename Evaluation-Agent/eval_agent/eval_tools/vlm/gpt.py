import base64
import requests
import os


class GPT:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.system_prompt = "You will receive an image and a question. Please start by answering the question with a simple 'Yes', 'No', or a brief answer. Afterward, provide a detailed explanation of how you arrived at your answer, including a rationale or description of the key details in the image that led to your conclusion. Ensure the evaluation is as precise and exacting as possible, scrutinizing the image thoroughly."
        self.input_tokens_count = 0
        self.output_tokens_count = 0
        

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    
    def update_tokens_count(self, response):
        self.input_tokens_count += response.json()["usage"]["prompt_tokens"]
        self.output_tokens_count += response.json()["usage"]["completion_tokens"]

    
    def show_usage(self):
        print(f"Total vlm input tokens used: {self.input_tokens_count}\nTotal vlm output tokens used: {self.output_tokens_count}")


    def predict(self, image_path, query):
        # Getting the base64 string
        base64_image = self.encode_image(image_path)

        payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
            "role": "system",
            "content": self.system_prompt
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        self.update_tokens_count(response)
        response_content = response.json()["choices"][0]["message"]["content"]

        return response_content
    
