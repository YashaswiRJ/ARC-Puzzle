import requests

api_key = "sk-21a4afdf403e4555831cd43ab52be697"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",  # or "deepseek-coder" depending on what you're using
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)

print(response.json())
