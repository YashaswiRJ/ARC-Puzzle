from langchain_ollama import OllamaLLM;
llm = OllamaLLM(model="deepseek-r1");
    
import os

import json
import dsl
import jsonToString

def load_conversation(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return [] 


def save_conversation(data_file_path, conversation):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    response_dir = os.path.join(current_dir, "llm_response_one_shot")
    os.makedirs(response_dir, exist_ok=True)
    base_name = os.path.basename(data_file_path)
    response_file = os.path.join(response_dir, f"{base_name}.response")
    with open(response_file, 'w') as file:
        file.write(conversation)

def one_shot_prompt(data_file_path):
    conversations = [];
    conversations.append('User: ' + dsl.fetch_dsl() + ',\n');
    print(conversations)
    conversations.append(jsonToString.extract_trees_from_json(f'{data_file_path}')); # Json input data
    print(conversations)

    one_shot_conversation = '\n'.join(conversations);
    response = llm.invoke(one_shot_conversation);

    conversations.append(f'Bot: {response}')
    string_response = '\n\n'.join(conversations);

    save_conversation(data_file_path, string_response);
    print(save_conversation)

import glob

def main():
    for filepath in glob.glob('./training/*.json'):
        print(f"Running for {filepath}")
        one_shot_prompt(filepath)

# def main():
#     one_shot_prompt('./data/868de0fa.json')
    
main()
    # conversations.append(f"User: {prompt}")
    # conversation_history = "\n".join(conversations)
    # response = llm.invoke("This was your previous conversation with the user, classifies as 'User' and 'Bot': " + conversation_history + ". Now please answer to the user's query, using the things learnt so far: " + prompt)
    # conversations.append(f"User: {prompt}")
    # conversations.append(f"Bot: {response}")
    
    # save_conversation(conversations)
    # print(response)