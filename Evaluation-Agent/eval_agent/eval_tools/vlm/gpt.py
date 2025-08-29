import base64, requests, os, mimetypes
from typing import Dict, Any

class GPT:
    def __init__(self):
        # 读取 key 与 base_url（优先环境变量）
        self.api_key  = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置")

        self.base_url = (os.getenv("OPENAI_BASE_URL")
                         or os.getenv("OPENAI_API_BASE")
                         or "https://api.chatanywhere.tech/v1").rstrip("/")

        # 模型名改成可配置；chatanywhere 常用 gpt-4o / gpt-3.5-turbo
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o").strip()

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.system_prompt = (
            "You will receive an image and a question. Start with a succinct answer "
            "('Yes'/'No' or brief text), then give a precise rationale with visual evidence."
        )

        self.input_tokens_count = 0
        self.output_tokens_count = 0
        self.timeout_s = float(os.getenv("OPENAI_HTTP_TIMEOUT", "60"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

    def _guess_mime(self, path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or "image/jpeg"

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def update_tokens_count(self, payload: Dict[str, Any]) -> None:
        try:
            u = payload.get("usage") or {}
            self.input_tokens_count  += int(u.get("prompt_tokens", 0) or 0)
            self.output_tokens_count += int(u.get("completion_tokens", 0) or 0)
        except Exception:
            pass

    def show_usage(self):
        print(f"Total vlm input tokens used: {self.input_tokens_count}\n"
              f"Total vlm output tokens used: {self.output_tokens_count}")

    def predict(self, image_path: str, query: str) -> str:
        b64 = self.encode_image(image_path)
        mime = self._guess_mime(image_path)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": query},
                     {"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}}]}
            ],
            "max_tokens": self.max_tokens
        }

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout_s)

        # 1) 先检查 HTTP
        if resp.status_code // 100 != 2:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

        # 2) 解析 JSON
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Invalid JSON: {resp.text[:800]}")

        # 3) usage 可选统计
        self.update_tokens_count(data)

        # 4) 错误返回统一抛出
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"API error: {data['error']}")

        # 5) 读取 choices
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing 'choices'. Got: {str(data)[:800]}")

        content = choices[0].get("message", {}).get("content", "")
        return (content or "").strip()
