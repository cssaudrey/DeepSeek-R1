import requests

# 基础初始化设置
base_url = "http://52.175.228.173:11434/api"
headers = {
    "Content-Type": "application/json"
}

def generate_completion_stream(prompt, model="deepseek-r1:1.5b"):
    url = f"{base_url}/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(url, headers=headers, json=data, stream=True)
    result = ""
    for line in response.iter_lines():
        if line:
            result += line.decode('utf-8')
    return result

# 示例调用
stream_completion = generate_completion_stream("上海在哪里？")
print("流式生成文本补全:", stream_completion)




