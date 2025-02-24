import openai
 
client = openai.Client(base_url="http://52.175.228.173:8123/v1", api_key="None")
 
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "user", "content": "100以内的质数有哪些？"},
    ],
    temperature=0.1,
    max_tokens=200,
)
print(response.choices[0].message.content)