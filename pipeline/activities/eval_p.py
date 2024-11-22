import os
import re
import argparse
import asyncio
import logging
import sys
import json
import io
import socket
import openai

import infiagent
from infiagent.utils import get_logger, upload_files, get_file_name_and_path
from infiagent.services.chat_complete_service import predict
logger = get_logger()

class UploadedFile(io.BytesIO):
    def __init__(self, path):
        with open(path, 'rb') as file:
            data = file.read()

        super().__init__(data)

        self.name = path.split("/")[-1]  # 获取文件名
        self.type = 'application/octet-stream'  # 或者其他适当的 MIME 类型
        self.size = len(data)

    def __repr__(self):
        return f"MyUploadedFile(name={self.name}, size={self.size}, type={self.type})"

    def __len__(self):

        return self.size

def _get_script_params():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm',
                            help='LLM Model for demo',
                            required=False, type=str)
        parser.add_argument('--api_key',
                            help='Open API token key.',
                            required=False, type=str)
        parser.add_argument('--api_base',
                            help="Base URL for the API",
                            required=False, type=str)
        parser.add_argument('--config_path',
                            help='Config path for demo',
                            default="configs/agent_configs/react_agent_llama_async.yaml",
                            required=False, type=str)

        args = parser.parse_args()

        return args
    except Exception as e:
        logger.error("Failed to get script input arguments: {}".format(str(e)), exc_info=True)

    return None

def extract_questions_and_concepts(file_path):
    # Read the content of the text file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to extract questions and concepts
    pattern = r'\Question{(.*?)}\s*\Concepts{(.*?)}'
    matches = re.findall(pattern, content, re.DOTALL)

    # Build a list of dictionaries containing the questions and concepts
    data = []
    for match in matches:
        question = match[0].strip()
        concepts = [concept.strip() for concept in match[1].split(',')]
        data.append({
            'question': question,
            'concepts': concepts
        })

    return data

def read_dicts_from_file(file_name):
    """
    Read a file with each line containing a JSON string representing a dictionary,
    and return a list of dictionaries.

    :param file_name: Name of the file to read from.
    :return: List of dictionaries.
    """
    dict_list = []
    with open(file_name, 'r') as file:
        for line in file:
            # Convert the JSON string back to a dictionary.
            dictionary = json.loads(line.rstrip('\n'))
            dict_list.append(dictionary)
    return dict_list

def read_questions(file_path):
    print(file_path)
    with open(file_path) as f:
        questions = json.load(f)

    return questions

def extract_data_from_folder(folder_path):

    print(f'folder_path {folder_path}')
    extracted_data = {}
    # Traverse the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.questions'):  # You can filter files based on their type
            file_path = os.path.join(folder_path, file_name)
            file_data = read_questions(file_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            extracted_data[file_name_without_extension] = file_data

    return extracted_data

async def process_question(q, model_name, args, table_path):
    """
    Process a single question asynchronously.
    """
    input_text = q['question']
    concepts = q['concepts']
    file_path = q['file_name']
    constraints = q['constraints']
    format = q['format']

    file_path = os.path.join(table_path, file_path)
    uploaded_file = UploadedFile(file_path)

    prompt = f"Question: {input_text}\n{constraints}\n"

    response = await predict(
        prompt=prompt,
        model_name=model_name,
        config_path=args.config_path,
        uploaded_files=[uploaded_file]
    )

    return {
        'id': q['id'],
        'input_text': prompt,
        'concepts': concepts,
        'file_path': file_path,
        'response': response,
        'format': format
    }

async def main():
    extracted_data = read_dicts_from_file('../examples/DA-Agent/data/da-dev-questions.jsonl')
    args = _get_script_params()

    model_name = getattr(args, "llm", None)
    open_ai_key = getattr(args, "api_key", None)
    open_ai_url = getattr(args, "api_base", None)

    print(f"model_name: {model_name}")
    print(f"open_ai_key: {open_ai_key}")
    print(f"open_ai_url: {open_ai_url}")
    if "OPEN_AI" in model_name:
        logger.info("setup open ai ")
        if open_ai_key:
            openai.api_key = open_ai_key
            os.environ["OPENAI_API_KEY"] = open_ai_key
        else:
            raise ValueError(
                "OPENAI_API_KEY is None, please provide open ai key to use open ai model. Adding '--api_key' to set it up")
        if open_ai_url:
            openai.api_base = open_ai_url
            os.environ["OPENAI_API_BASE"] = open_ai_url
        else:
            raise ValueError(
                "OPENAI_API_BASE is None, please provide open ai key to use open ai model. Adding '--api_key' to set it up")

        # 获取 'openai' 的 logger
        openai_logger = logging.getLogger('openai')
        # 设置日志级别为 'WARNING'，这样 'INFO' 级别的日志就不会被打印了
        openai_logger.setLevel(logging.WARNING)
    else:
        logger.info("use local model ")

    table_path = '../examples/DA-Agent/data/da-dev-tables'

    results = []
    semaphore = asyncio.Semaphore(30)  # 限制并发量，防止地址冲突或资源占用

    async def limited_process_question(q):
        async with semaphore:
            return await process_question(q, model_name, args, table_path)

    # 使用 asyncio.gather 分批次处理每个问题，每 10 个保存一次结果
    for i in range(120, len(extracted_data), 10):
        tasks = [limited_process_question(q) for q in extracted_data[i:i + 10]]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # 每 10 个问题保存一次结果
        with open('results_{}.json'.format(model_name), 'w') as outfile:
            json.dump(results, outfile, indent=4)

    # 最终保存结果
    with open('results_{}.json'.format(model_name), 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == '__main__':
    # 解决端口冲突的常见问题，通过捕获 OSError 并重试来获取可用端口
    for _ in range(30):
        try:
            asyncio.run(main())
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning("Address already in use, retrying...")
                continue
            else:
                raise
