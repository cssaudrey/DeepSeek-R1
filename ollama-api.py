#Ollama API调用示例

import requests

# 基础初始化设置
base_url = "http://52.175.228.173:11434/api"
headers = {
    "Content-Type": "application/json"
}


def generate_completion(prompt, model="deepseek-r1:1.5b"):
    url = f"{base_url}/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get('response', '')

# 示例调用
completion = generate_completion("介绍一下人工智能。")
print("生成文本补全:", completion)

