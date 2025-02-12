# GeminiMixLite

本项目是[GeminiMixSuper](https://github.com/lioensky/GeminiMixSuper)的**简易版**

>本项目是一个混合 AI 代理服务器，结合了 deepseek-r1 模型的思考能力和 gemini 模型的强大语料库与多模态能力。它可以接收类似 **OpenAI API 格式**的请求，支持文本对话，并返回流式响应。

#与完整版[GeminiMixSuper](https://github.com/lioensky/GeminiMixSuper)的差别

+ +对Gemini API无高并发需求，无动态负载需求，仅需**中转**即可。

- -无多模态支持，无智能搜索功能。

## 安装

1. **克隆仓库：**

    ```bash
    git clone https://github.com/shanchuan001/GeminiMixLite.git
    cd GeminiMixLite
    ```

2. **安装依赖：**

    ```bash
    pip install -r requirements.txt
    ```

3. **配置环境变量** (见下文 "环境变量" 部分)。

5. **启动服务：**

    ```bash
    python GeminiMIXR1-Lite.py
    ```

    服务默认运行在 `http://localhost:5000`。

## 使用

### 1. 发送聊天请求 (流式)

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "messages": [
      {"role": "user", "content": "请写一个快速排序算法。"}
    ],
    "stream": true
  }'
```
### 2. API 参数
  本服务支持从前端输入以下 `OpenAI 风格的 API 参数`（注意：这些参数仅影响“输出模型”的行为, **“推理模型”参数请在程序内修改** ）：
  - messages **(必需)**: 聊天记录，格式为 ` [{"role": "user", "content": "..." }, {"role": "assistant", "content": "..." }, ...]。 `
  - stream (可选): 是否开启流式响应。默认为 false。
  - max_tokens (可选): 生成响应的最大 token 数。
  - temperature (可选): 控制生成文本的随机性。值越高，文本越随机。
  - top_p (可选): 控制生成文本的多样性。
  - top_k (可选): 控制生成文本的多样性。
  - frequency_penalty (可选): 控制生成文本中重复词的惩罚。
  - n (可选): 控制生成文本的几个备选响应, 默认为一个。
  - stop (可选): 指定停止生成的字符串序列。

### 3. 在[Cherry Studio](https://github.com/CherryHQ/cherry-studio)中使用

 - 在`设置`-`模型服务`中添加**供应商类型**为`openai`的服务
 - 填入**你设置的**密钥(默认`123`)与地址(例如`http://localhost:5000`)
 - 在管理中添加模型`GeminiMIXR1`(或者其他自定义名称)

## 依赖
```bash
flask~=2.0
requests~=2.26
```

## 环境变量
为了使服务的运行，您需设置以下环境变量，或在**程序内**填入默认值：

 - INFERENCE_API_URL: 推理模型（例如 DeepSeek-R1）的 API 地址。 示例：https://api.example.com/v1
 - INFERENCE_API_KEY: 推理模型的 API 密钥。
 - OUTPUT_API_URL: 输出模型（例如 Gemini）的 API 地址。 示例：https://api.another-example.com/v1
 - OUTPUT_API_KEY: 输出模型的 API 密钥。
 - API_KEY: 访问此聊天完成服务的 API 密钥。 如果设置了此变量，客户端请求必须在请求头中包含 Authorization: Bearer YOUR_API_KEY。
   
安全提示： 请**妥善保管**您的 API 密钥，不要将其直接暴露在客户端代码或公开仓库中。
