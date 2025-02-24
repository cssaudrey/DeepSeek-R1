from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
 
client = ChatCompletionsClient(endpoint="https://DeepSeek-R1-Oli.eastus.models.ai.azure.com", credential=AzureKeyCredential("Kh1R7sfzcmHXLKmDvXh0kPgxjz64Esoj"))
 
response = client.complete(
    model= "DeepSeek-R1",
    stream=True,
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("Give me 5 good reasons why I should exercise every day."),
    ],
)
 
# 遍历response对象中的每一个update
for update in response:
    
    if update.choices and len(update.choices) > 0:
       
        content = update.choices[0].delta.content or ""
       
        print(content, end="", flush=True)
 
client.close()