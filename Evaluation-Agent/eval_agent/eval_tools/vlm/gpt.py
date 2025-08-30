import base64, requests, os, mimetypes
from typing import Dict, Any
class GPT:
    def __init__(self):
        # 从环境变量读取 OpenAI API Key
        
        # 获取 OPENAI_API_KEY，如果环境变量不存在则用空字符串，去掉前后空格
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set or empty")
            
        # 获取环境变量 OPENAI_MODEL，如果不存在则用默认值 "gpt-4o"，并去掉前后空格
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
        if not self.model:
            raise ValueError("Environment variable OPENAI_MODEL is not set or empty")

        # 仅当使用第三方或自建 API 时才使用 base_url，否则 None
        self.base_url = None
        third_party_url = (os.getenv("OPENAI_BASE_URL", "") or os.getenv("OPENAI_API_BASE", "")).strip().rstrip("/")
        if third_party_url:
            self.base_url = third_party_url
                # 使用示例
        if self.base_url:
            print(f"Using third-party API at {self.base_url}")
        else:
            print("Using official OpenAI API")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.base_url}"  # Bearer token 授权
        }

        # 系统提示，用于指导模型如何回答问题
        self.system_prompt = (
            "You will receive an image and a question. Please start by answering the question "
            "with a simple 'Yes', 'No', or a brief answer. Afterward, provide a detailed explanation "
            "of how you arrived at your answer, including a rationale or description of the key details "
            "in the image that led to your conclusion. Ensure the evaluation is as precise and exacting as possible, "
            "scrutinizing the image thoroughly."
        )
        
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
        
        # token 统计
        self.input_tokens_count = 0
        self.output_tokens_count = 0

    def encode_image(self, image_path):
        """
        输入: image_path -> 本地图片路径
        输出: base64 string -> 可用于直接发送到 API
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _guess_mime(self, path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or "image/jpeg"

    def predict(self, image_path, query):
        """
        核心方法: 将图片 + 文本问题发送给 VLM GPT API
        输入:
            - image_path: 本地图片路径
            - query: 用户问题字符串
        输出:
            - response_content: 模型回答字符串
        """
        # 1️⃣ 图片编码成 base64
        base64_image = self.encode_image(image_path)
        mime = self._guess_mime(image_path)

        # 2️⃣ 构造 payload，符合 OpenAI Chat Completions VLM 格式
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": self.max_tokens
        }

        # 3️⃣ 调用 API
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self.headers, json=payload)

        # 1) 先检查 HTTP
        if resp.status_code // 100 != 2:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

        # 2) 解析 JSON
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Invalid JSON: {resp.text[:800]}")

        # 3) 错误返回统一抛出
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"API error: {data['error']}")

        # 4) 读取 choices
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing 'choices'. Got: {str(data)[:800]}")

        content = choices[0].get("message", {}).get("content", "")
        return (content or "").strip()