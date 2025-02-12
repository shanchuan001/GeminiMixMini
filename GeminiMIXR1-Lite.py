from flask import Flask, request, jsonify, stream_with_context, Response
import time
import json
import requests
import os
import uuid

app = Flask(__name__)

# --- 环境变量 --- ##只支持 openai 格式api
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "从环境变量获取,或者在这里填上推理模型的API URL")
INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY", "从环境变量获取,或者在这里填上推理模型的API KEY")
OUTPUT_API_URL = os.environ.get("OUTPUT_API_URL", "从环境变量获取,或者在这里填上输出模型的API URL")
OUTPUT_API_KEY = os.environ.get("OUTPUT_API_KEY", "从环境变量获取,或者在这里填上输出模型的API KEY")

# --- API 密钥 ---
API_KEY = 123  #  建议也从环境变量获取，并提供默认值
# API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY")


# --- 输出模型名称 ---
MIXED_MODEL_NAME = "GeminiMIXR1-lite"


if not INFERENCE_API_KEY:
    print("警告：未设置 'INFERENCE_API_KEY'。推理模型将无法工作。")
if not OUTPUT_API_KEY:
    print("警告：未设置 'OUTPUT_API_KEY'。输出模型将无法工作。")

# --- 使用模型名称 ---
INFERENCE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # 可以根据需要更改 "deepseek-ai/DeepSeek-R1"
OUTPUT_MODEL_NAME = "gemini-exp-1206"  # 可以根据需要更改


# --- 提示词 (修改：内置系统提示词) ---
INFERENCE_BUILTIN_SYSTEM_PROMPT = """你是一个前置思考系统，一个专业的推理引擎。你的任务是接收用户输入的信息，并将其分解为清晰的、结构化的思维链，以便后续的模型进行处理和输出最终回复。
具体步骤：

1.  接收输入：获取用户提供的问题。
2.  问题解构：将问题分解为更小、更具体的子问题。并挨个考虑。
3.  信息评估：评估输入信息的准确性和可靠性，并指出可能存在的疑问或不确定性。
4.  构建思维链：以编号列表的形式，逐步展示你的推理过程。每个步骤应简洁明了，并清晰地表达关键信息。

约束条件：

*   **只进行推理，不进行具体计算，也不提出结论或者答案**
*   保持客观中立。
*   所有分析应基于相应原理。
*   输出应简洁、清晰，避免冗余信息。

输出格式：
... (CoT内容，不包含任何与推理无关的描述性内容) ...
---以上是辅助思考内容---

请严格按照上述指示执行。绝对禁止使用Markdown或其他复杂格式。
"""

OUTPUT_BUILTIN_SYSTEM_PROMPT = """
#前置思考系统已完成思考，结合辅助思考内容和**你自己的思考**，开始你的正式输出。辅助思考的内容在上一个assistant的输出中。
##其内容用户不可见只有Model也就是你可见。
##前置思考系统的知识库相对较为陈旧仅停留在2023年4月，很多新生代知识还未接触，而实际现实时间已经抵达2025年初。所以，有些内容需要你自行辨别其思考内容的正确性。
##请勿向用户过多谈起前置思考系统提及的这些技术内容。
"""

# --- 模型参数 (仅推理模型) ---
DEFAULT_MAX_TOKENS = 4096
DEFAULT_STOP = ["null"]
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.5
DEFAULT_TOP_K = 50
DEFAULT_FREQUENCY_PENALTY = 0.2
DEFAULT_N = 1

# --- API 密钥验证装饰器 (Express 风格) ---
def require_api_key(func):
    """装饰器：要求请求必须包含有效的 API 密钥。"""
    def wrapper(*args, **kwargs):
        # 兼容 Express 和 Flask 的请求头获取方式
        provided_key = request.headers.get("Authorization")
        if provided_key:
             provided_key = provided_key.replace("Bearer ", "").strip()  # 从请求头获取密钥, 并处理 Bearer Token

        if not provided_key:  # 优先检查是否提供了密钥
            return jsonify(create_error_response("未提供 API 密钥。")), 401

        if provided_key != str(API_KEY):  # 使用 str() 转换，确保类型一致
            return jsonify(create_error_response("无效的 API 密钥。")), 401

        return func(*args, **kwargs)  # 调用原函数
    return wrapper


# --- API 客户端函数 ---
def call_inference_model(messages):
    """调用推理模型 API，强制使用程序内定义的参数。"""
    if not INFERENCE_API_KEY:
        return {"error": "未配置推理模型 API 密钥。"}

    api_endpoint = f"{INFERENCE_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {INFERENCE_API_KEY}",
        "Content-Type": "application/json"
    }

    inference_messages = []
    system_content = INFERENCE_BUILTIN_SYSTEM_PROMPT
    inference_messages.append({"role": "system", "content": system_content})

    for message in messages:
        if message['role'] == 'user':
            inference_messages.append(message)
        elif message['role'] == 'assistant':
            inference_messages.append({'role':'assistant', 'content': ''})

    # 使用程序内部定义的参数
    model_params = {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stop": DEFAULT_STOP,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "n": DEFAULT_N
    }

    payload = {
        "messages": inference_messages,
        "model": INFERENCE_MODEL_NAME,
        "stream": True,
        **model_params  # 使用固定的 model_params
    }
    print("发送给推理模型的数据：", json.dumps(payload, indent=4, ensure_ascii=False))

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8')
                    json_str = decoded_line.replace('data: ', '', 1).strip()

                    if json_str:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0]['delta']
                            delta_content = delta.get('reasoning_content') or delta.get('content', '')
                            if delta_content:
                                yield delta_content

                except json.JSONDecodeError as e:
                    print(f"JSON解码错误: {line.decode('utf-8', 'ignore')}")
                    print(f"解码错误详情: {e}")
                except Exception as e:
                    print(f"其他错误: {e}")

    except requests.exceptions.RequestException as e:
        yield {"error": f"调用推理模型 API 时出错：{e}"}
    except Exception as e:
        yield {"error": f"处理推理模型响应时出错：{e}"}



def call_output_model(full_inference_reasoning, messages, model_params, output_model_name):
    """调用输出模型 API，使用用户提供的参数或默认参数。"""
    if not OUTPUT_API_KEY:
        return {"error": "未配置输出模型 API 密钥。"}

    api_endpoint = f"{OUTPUT_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OUTPUT_API_KEY}",
        "Content-Type": "application/json"
    }

    output_messages = []
    user_system_prompt = None

    for message in messages:
        if message['role'] == 'system':
            user_system_prompt = message['content']
            break

    if user_system_prompt:
        output_messages.append({"role": "system", "content": user_system_prompt})

    for message in messages:
        if message['role'] in ('user', 'assistant'):
            output_messages.append(message)

    output_messages.append({"role": "assistant", "content": full_inference_reasoning})
    output_messages.append({"role": "user", "content": OUTPUT_BUILTIN_SYSTEM_PROMPT})

    payload = {
        "model": output_model_name,
        "messages": output_messages,
        "stream": True,
        **model_params  # 使用传入的 model_params (包含用户提供的或默认的)
    }

    print("发送给输出模型的数据：", json.dumps(payload, indent=4, ensure_ascii=False))

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.strip() == "data: [DONE]":
                        continue

                    json_str = decoded_line.replace('data: ', '', 1).strip()
                    if json_str:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and chunk['choices']:
                            delta_content = chunk['choices'][0]['delta'].get('content', '')
                            if delta_content:
                                yield delta_content

                except json.JSONDecodeError as e:
                    print(f"JSON解码错误: {line.decode('utf-8', 'ignore')}")
                    print(f"解码错误详情: {e}")
                except Exception as e:
                    print(f"其他错误: {e}")

    except requests.exceptions.RequestException as e:
        yield {"error": f"调用 {output_model_name} API 时出错：{e}"}
    except Exception as e:
        yield {"error": f"处理{output_model_name}响应时出错：{e}"}


def process_request_stream(messages, model_params, stream):
    """处理单个请求，返回流式响应生成器。"""

    full_inference_reasoning = ""
    inference_reason_generator = call_inference_model(messages)

    try:
        for chunk in inference_reason_generator:
            if isinstance(chunk, dict) and "error" in chunk:
                yield from generate_error_stream(chunk["error"])
                return
            reasoning_chunk = chunk
            full_inference_reasoning += reasoning_chunk
            stream_chunk = create_stream_chunk_delta(reasoning_chunk, is_reasoning=True)
            yield f"data: {json.dumps(stream_chunk)}\n\n"
            time.sleep(0.02)

    except Exception as e:
        yield from generate_error_stream(f"处理推理模型推理流时出错: {e}")
        return

    output_model_generator = call_output_model(full_inference_reasoning, messages, model_params, OUTPUT_MODEL_NAME)

    try:
        for chunk in output_model_generator:
            if isinstance(chunk, dict) and "error" in chunk:
                yield from generate_error_stream(chunk["error"])
                return
            response_chunk = chunk
            stream_chunk = create_stream_chunk_delta(response_chunk, is_reasoning=False)
            yield f"data: {json.dumps(stream_chunk)}\n\n"
            time.sleep(0.02)

    except Exception as e:
        yield from generate_error_stream(f"处理输出模型流时出错: {e}")
        return

    finish_chunk = create_stream_chunk_finish()
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# --- API 端点 ---
@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key  # 应用 API 密钥验证装饰器
def chat_completions():
    """处理聊天完成请求，直接处理流式或非流式请求。"""
    data = request.get_json()
    messages = data.get('messages', [])
    stream = data.get('stream', False)

    # 提取模型控制参数 (用于 Gemini)
    model_params = {
        "max_tokens": data.get("max_tokens"),
        "stop": data.get("stop"),
        "temperature": data.get("temperature"),
        "top_p": data.get("top_p"),
        "top_k": data.get("top_k"),
        "frequency_penalty": data.get("frequency_penalty"),
        "n": data.get("n")
    }
     # 验证 stop 列表中的每个元素是否为字符串
    if model_params["stop"] and not all(isinstance(item, str) for item in model_params["stop"]):
        return jsonify(create_error_response("`stop`参数必须是字符串列表")), 400

    if stream:
        return Response(stream_with_context(process_request_stream(messages, model_params, stream)), mimetype='text/event-stream')
    else:
        full_inference_reasoning = ""
        inference_reason_gen = call_inference_model(messages)  # 移除 model_params
        try:
            for chunk in inference_reason_gen:
                if isinstance(chunk, dict) and "error" in chunk:
                    return jsonify(create_error_response(chunk["error"])), 500
                full_inference_reasoning += chunk
        except Exception as e:
            return jsonify(create_error_response(f"处理推理模型推理流时出错: {e}")), 500

        output_model_response_content = ""
        output_model_gen = call_output_model(full_inference_reasoning, messages, model_params, OUTPUT_MODEL_NAME)
        try:
            for chunk in output_model_gen:
                if isinstance(chunk, dict) and "error" in chunk:
                     return jsonify(create_error_response(chunk["error"])), 500
                output_model_response_content += chunk
        except Exception as e:
            return jsonify(create_error_response(f"处理输出模型流时出错: {e}")), 500

        assistant_message = {'role': 'assistant', 'content': output_model_response_content,
                             'reasoning_content': full_inference_reasoning}
        response_data = create_completion_response(assistant_message, OUTPUT_MODEL_NAME)
        return jsonify(response_data)


@app.route('/v1/models', methods=['GET'], endpoint='list_models')
@app.route('/models', methods=['GET'], endpoint='list_models')
@require_api_key  # 对/models 端点应用 API 密钥验证
def list_models():
    """列出可用模型。"""
    model_list = {
        "object": "list",
        "data": [
            {"id": MIXED_MODEL_NAME, "object": "model", "owned_by": "mixmodel"}
        ]
    }
    return jsonify(model_list)



# --- 响应助手函数 ---
def create_completion_response(assistant_message, model_name):
    """创建标准的 OpenAI 风格的完成响应，包含 reasoning_content。"""
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "message": assistant_message,
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 100, # 占位符值
            "completion_tokens": 50, # 占位符值
            "total_tokens": 150 # 占位符值
        }
    }

def create_stream_chunk_start(model_name):
    """创建流式传输的初始块。"""
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "delta": {},
                "index": 0,
                "finish_reason": None
            }
        ]
    }

def create_stream_chunk_delta(content, is_reasoning=False):
    """创建用于流式传输内容的增量块，包含 reasoning_content 或 content。"""
    delta_content = {}
    if is_reasoning:
        delta_content["reasoning_content"] = content
    else:
        delta_content["content"] = content

    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": OUTPUT_MODEL_NAME,
        "choices": [
            {
                "delta": delta_content,
                "index": 0,
                "finish_reason": None
            }
        ]
    }


def create_stream_chunk_finish():
    """创建流式传输的最终块。"""
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": OUTPUT_MODEL_NAME,
        "choices": [
            {
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }

def create_error_response(error_message):
    """创建 OpenAI 风格的错误响应。"""
    return {
        "error": {
            "message": error_message,
            "type": "APIError",
            "param": None,
            "code": None
        }
    }

def generate_error_stream(error_message):
    """生成错误信息的流。"""
    error_chunk = {
        "error": {
            "message": error_message,
            "type": "APIError",
            "param": None,
            "code": None
        }
    }
    yield f"data: {json.dumps(error_chunk)}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == '__main__':
    app.run(debug=True, port=5000)
