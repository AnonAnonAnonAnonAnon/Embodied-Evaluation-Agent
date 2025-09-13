import os
import requests

def query_lightrag(query: str, 
                   top_k: int = 15, 
                   kg_top_k: int = 50, 
                   chunk_top_k: int = 30, 
                   max_entity_tokens: int = 8000, 
                   max_relation_tokens: int = 10000, 
                   max_total_tokens: int = 35000, 
                   user_prompt: str = '', 
                   enable_rerank: bool = True, 
                   only_need_context: bool = True, 
                   only_need_prompt: bool = False, 
                   stream_response: bool = True, 
                   endpoint: str = '/query', 
                   base_url: str = "http://202.38.247.58"):

    headers = {"Content-Type": "application/json"}
    if os.getenv("LIGHTRAG_API_KEY"):  # check if API Key is set in environment variable
        headers["Authorization"] = f"Bearer {os.getenv('LIGHTRAG_API_KEY')}"
    
    # Prepare the payload with all configurable parameters
    payload = {
        "query": query,
        "top_k": top_k,
        "kg_top_k": kg_top_k,
        "chunk_top_k": chunk_top_k,
        "max_entity_tokens": max_entity_tokens,
        "max_relation_tokens": max_relation_tokens,
        "max_total_tokens": max_total_tokens,
        "user_prompt": user_prompt,
        "enable_rerank": enable_rerank,
        "only_need_context": only_need_context,
        "only_need_prompt": only_need_prompt,
        "stream_response": stream_response
    }
    
    # Make the POST request to the specified endpoint
    resp = requests.post(f"{base_url}{endpoint}", headers=headers, json=payload, timeout=60)
    
    # Raise an exception if the request was unsuccessful
    resp.raise_for_status()
    
    return resp.json()

# Example usage:
response = query_lightrag(
    query="RoboTwin是什么？", 
    top_k=8,
    kg_top_k=40, 
    chunk_top_k=20,
    max_entity_tokens=6000, 
    max_relation_tokens=8000, 
    max_total_tokens=30000,
    user_prompt="Custom prompt here",
    enable_rerank=True,
    stream_response=True,
    only_need_context=True,
)

print(response)