import argparse
import json

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="deepseek-r1") 

conversations = []

file_path = 'rpro.txt'
history_file = 'output.json'

def load_conversation():
    try:
        with open(history_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  
    
def save_conversation(conversations):
    with open(history_file, 'w') as file:
        json.dump(conversations, file)


def read_prompt_from_file():
    try:
        with open(file_path, 'r') as file:
            prompt = file.read().strip() 
        open(file_path, 'w').close()  
        return main(prompt)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def main(prompt):
    
    conversations = load_conversation()

    conversations.append(f"User: {prompt}")
    conversation_history = "\n".join(conversations)
    response = llm.invoke("This was your previous conversation with the user, classifies as 'User' and 'Bot': " + conversation_history + ". Now please answer to the user's query, using the things learnt so far: " + prompt)
    conversations.append(f"User: {prompt}")
    conversations.append(f"Bot: {response}")
    
    save_conversation(conversations)
    print(response)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Invoke Llama model with a custom prompt.")
    parser.add_argument('rpro', type=str, help="The prompt to send to the model")
    return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_arguments()
#     main(args.prompt)

if __name__ == '__main__':
    read_prompt_from_file()