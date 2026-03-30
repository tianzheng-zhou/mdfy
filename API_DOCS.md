# mdfy API 文档

PDF -> Markdown AI 转换服务，RESTful API v1。

这份文档适合直接发给调用方使用。服务启动后，默认基于 Web 服务地址提供 API，例如：

- Web 页面: http://127.0.0.1:23504
- API Base URL: http://127.0.0.1:23504/api/v1
- 在线 HTML 文档: http://127.0.0.1:23504/api/docs
- JSON 文档: http://127.0.0.1:23504/api/v1/docs

## 1. 概览

API 采用异步任务模式：

1. 上传 PDF，创建任务。
2. 轮询任务状态，或订阅 SSE 进度流。
3. 任务完成后读取 Markdown、图片列表，或直接下载 ZIP / Markdown 文件。

统一 JSON 响应格式如下：

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

- success: 请求是否成功
- data: 成功时返回的业务数据
- error: 失败时返回错误信息

## 2. 认证

如果服务端设置了环境变量 MDFY_API_KEY，则除健康检查和文档接口外，其余 API 都需要认证。

支持两种传参方式：

### 2.1 Header

```http
X-API-Key: your-key-here
```

### 2.2 Query 参数

```text
?api_key=your-key-here
```

如果服务端没有设置 MDFY_API_KEY，则 API 默认不强制认证，适合本地或内网场景。

## 3. 典型调用流程

### 3.1 创建转换任务

调用 POST /convert，上传 PDF 文件。

### 3.2 查询任务进度

二选一：

- 轮询 GET /tasks/{task_id}
- 订阅 GET /tasks/{task_id}/progress 的 SSE 日志流

### 3.3 获取结果

任务状态为 done 后，可使用：

- GET /tasks/{task_id}/result 获取完整结果
- GET /tasks/{task_id}/result/markdown 只取 Markdown 文本
- GET /tasks/{task_id}/images 获取图片列表
- GET /tasks/{task_id}/download 下载 ZIP
- GET /tasks/{task_id}/download/markdown 下载 Markdown 文件

## 4. 任务状态

| 状态 | 含义 |
| --- | --- |
| pending | 任务已创建，等待处理 |
| converting | 正在转换 |
| done | 转换完成 |
| error | 转换失败 |

## 5. 接口清单

| 方法 | 路径 | 是否需认证 | 说明 |
| --- | --- | --- | --- |
| GET | /health | 否 | 健康检查 |
| GET | /models | 是 | 获取当前可用模型列表 |
| POST | /convert | 是 | 上传 PDF 并创建异步转换任务 |
| GET | /tasks/{task_id} | 是 | 查询任务状态 |
| GET | /tasks/{task_id}/progress | 是 | 订阅 SSE 进度流 |
| GET | /tasks/{task_id}/result | 是 | 获取 Markdown 和图片列表 |
| GET | /tasks/{task_id}/result/markdown | 是 | 仅获取 Markdown 文本 |
| GET | /tasks/{task_id}/images | 是 | 获取图片文件名列表 |
| GET | /tasks/{task_id}/images/{filename} | 是 | 获取指定图片文件 |
| GET | /tasks/{task_id}/download | 是 | 下载 Markdown + 图片 ZIP |
| GET | /tasks/{task_id}/download/markdown | 是 | 下载 Markdown 文件 |
| GET | /docs | 否 | JSON 版 API 文档 |

说明：

- 上表路径均相对于 /api/v1。
- /docs 的完整地址是 /api/v1/docs。

## 6. 接口详情

### 6.1 健康检查

请求：

```http
GET /api/v1/health
```

返回示例：

```json
{
  "success": true,
  "data": {
    "status": "ok",
    "timestamp": 1743235200.0
  },
  "error": null
}
```

### 6.2 获取模型列表

请求：

```http
GET /api/v1/models
```

返回说明：

- models: 当前服务可用模型列表
- default: 默认模型

返回示例：

```json
{
  "success": true,
  "data": {
    "models": [
      "qwen3.5-flash",
      "qwen3.5-plus"
    ],
    "default": "qwen3.5-plus"
  },
  "error": null
}
```

### 6.3 创建转换任务

请求：

```http
POST /api/v1/convert
Content-Type: multipart/form-data
```

表单参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| file | File | 是 | 要转换的 PDF 文件 |
| model | String | 否 | 模型名称；建议先通过 /models 获取当前可用列表 |

成功返回状态码：202 Accepted

返回示例：

```json
{
  "success": true,
  "data": {
    "task_id": "20260329_120000_abc123def456",
    "pdf_name": "example.pdf",
    "model": "qwen3.5-plus",
    "status": "pending"
  },
  "error": null
}
```

常见错误：

- 缺少 file 字段
- 上传的不是 PDF 文件
- model 不在服务端支持列表中

### 6.4 查询任务状态

请求：

```http
GET /api/v1/tasks/{task_id}
```

返回字段：

| 字段 | 说明 |
| --- | --- |
| task_id | 任务 ID |
| pdf_name | 原始 PDF 文件名 |
| model | 本次任务使用的模型 |
| status | 任务状态 |
| elapsed_seconds | 已耗时，单位秒；任务未开始时可能为 null |
| error | 失败时的错误信息 |

返回示例：

```json
{
  "success": true,
  "data": {
    "task_id": "20260329_120000_abc123def456",
    "pdf_name": "example.pdf",
    "model": "qwen3.5-plus",
    "status": "converting",
    "elapsed_seconds": 12.3,
    "error": null
  },
  "error": null
}
```

### 6.5 订阅 SSE 进度流

请求：

```http
GET /api/v1/tasks/{task_id}/progress
Accept: text/event-stream
```

说明：

- 普通日志以默认 SSE message 事件发送，格式为 data: 日志文本
- 转换完成时发送 done 事件，数据格式为 JSON
- 转换失败时发送 error 事件，数据格式为 JSON

事件示例：

```text
data: 正在分析 PDF 类型

data: 第 1 页转换完成

event: done
data: {"elapsed": 18.4}
```

失败事件示例：

```text
event: error
data: {"error": "模型调用失败"}
```

### 6.6 获取完整结果

请求：

```http
GET /api/v1/tasks/{task_id}/result
```

只有任务状态为 done 时可调用，否则返回 409。

返回字段：

| 字段 | 说明 |
| --- | --- |
| task_id | 任务 ID |
| markdown | 转换后的 Markdown 全文 |
| images | 输出目录中的 PNG 图片文件名列表 |
| image_count | 图片数量 |

返回示例：

```json
{
  "success": true,
  "data": {
    "task_id": "20260329_120000_abc123def456",
    "markdown": "# 文档标题\n\n正文内容...\n\n![图片](images/page0_img1.png)",
    "images": [
      "page0_img1.png",
      "page1_img1.png"
    ],
    "image_count": 2
  },
  "error": null
}
```

### 6.7 仅获取 Markdown 文本

请求：

```http
GET /api/v1/tasks/{task_id}/result/markdown
```

返回示例：

```json
{
  "success": true,
  "data": {
    "markdown": "# 文档标题\n\n正文内容..."
  },
  "error": null
}
```

### 6.8 获取图片列表

请求：

```http
GET /api/v1/tasks/{task_id}/images
```

返回字段：

| 字段 | 说明 |
| --- | --- |
| images | PNG 文件名列表 |
| count | 图片数量 |
| base_url | 获取单张图片时的路径前缀 |

返回示例：

```json
{
  "success": true,
  "data": {
    "images": [
      "page0_img1.png",
      "page1_img1.png"
    ],
    "count": 2,
    "base_url": "/api/v1/tasks/20260329_120000_abc123def456/images/"
  },
  "error": null
}
```

### 6.9 获取单张图片

请求：

```http
GET /api/v1/tasks/{task_id}/images/{filename}
```

返回：

- 成功时直接返回图片二进制内容
- Content-Type 由 Flask 根据文件类型自动设置，当前图片输出为 PNG

### 6.10 下载 ZIP

请求：

```http
GET /api/v1/tasks/{task_id}/download
```

返回：

- 一个 ZIP 文件
- ZIP 内包含 Markdown 文件和 images 目录

### 6.11 下载 Markdown 文件

请求：

```http
GET /api/v1/tasks/{task_id}/download/markdown
```

返回：

- 直接下载 .md 文件

### 6.12 获取 JSON 文档

请求：

```http
GET /api/v1/docs
```

说明：

- 返回机器可读的 JSON 文档
- 适合做接口探测、自动化集成或二次封装

## 7. 状态码

| 状态码 | 含义 |
| --- | --- |
| 200 | 请求成功 |
| 202 | 任务已创建，后台异步处理中 |
| 400 | 参数错误 |
| 401 | API Key 缺失或错误 |
| 404 | 任务、图片或文件不存在 |
| 409 | 任务尚未完成，当前还不能取结果 |

## 8. 错误响应示例

### 8.1 认证失败

```json
{
  "success": false,
  "data": null,
  "error": "认证失败：请提供有效的 API Key（Header: X-API-Key 或 Query: api_key）"
}
```

### 8.2 任务不存在

```json
{
  "success": false,
  "data": null,
  "error": "任务不存在"
}
```

### 8.3 任务未完成

```json
{
  "success": false,
  "data": null,
  "error": "任务尚未完成，当前状态: converting"
}
```

## 9. 调用示例

### 9.1 cURL

```bash
# 1. 上传 PDF，创建任务
curl -X POST http://127.0.0.1:23504/api/v1/convert \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf" \
  -F "model=qwen3.5-plus"

# 2. 查询任务状态
curl http://127.0.0.1:23504/api/v1/tasks/TASK_ID \
  -H "X-API-Key: your-key"

# 3. 获取完整结果
curl http://127.0.0.1:23504/api/v1/tasks/TASK_ID/result \
  -H "X-API-Key: your-key"

# 4. 下载 ZIP
curl -OJ http://127.0.0.1:23504/api/v1/tasks/TASK_ID/download \
  -H "X-API-Key: your-key"
```

如果服务端没有启用 MDFY_API_KEY，可以去掉认证头。

### 9.2 Python

```python
import time
import requests

BASE = "http://127.0.0.1:23504/api/v1"
HEADERS = {"X-API-Key": "your-key"}

with open("document.pdf", "rb") as file_obj:
    response = requests.post(
        f"{BASE}/convert",
        headers=HEADERS,
        files={"file": file_obj},
        data={"model": "qwen3.5-plus"},
    )
    response.raise_for_status()

task_id = response.json()["data"]["task_id"]
print("task_id:", task_id)

while True:
    status_response = requests.get(f"{BASE}/tasks/{task_id}", headers=HEADERS)
    status_response.raise_for_status()
    task = status_response.json()["data"]
    print("status:", task["status"], "elapsed:", task["elapsed_seconds"])

    if task["status"] == "done":
        break
    if task["status"] == "error":
        raise RuntimeError(task["error"])

    time.sleep(3)

result_response = requests.get(f"{BASE}/tasks/{task_id}/result", headers=HEADERS)
result_response.raise_for_status()
result = result_response.json()["data"]

print("image_count:", result["image_count"])
print(result["markdown"][:500])
```

### 9.3 JavaScript

```javascript
const fs = require("fs");
const FormData = require("form-data");
const fetch = require("node-fetch");

const BASE = "http://127.0.0.1:23504/api/v1";
const HEADERS = { "X-API-Key": "your-key" };

async function convert(pdfPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(pdfPath));
  form.append("model", "qwen3.5-plus");

  const uploadResponse = await fetch(`${BASE}/convert`, {
    method: "POST",
    headers: HEADERS,
    body: form,
  });

  if (!uploadResponse.ok) {
    throw new Error(`upload failed: ${uploadResponse.status}`);
  }

  const uploadData = await uploadResponse.json();
  const taskId = uploadData.data.task_id;

  while (true) {
    const statusResponse = await fetch(`${BASE}/tasks/${taskId}`, {
      headers: HEADERS,
    });

    if (!statusResponse.ok) {
      throw new Error(`status failed: ${statusResponse.status}`);
    }

    const statusData = await statusResponse.json();
    const task = statusData.data;

    if (task.status === "done") {
      break;
    }

    if (task.status === "error") {
      throw new Error(task.error);
    }

    await new Promise((resolve) => setTimeout(resolve, 3000));
  }

  const resultResponse = await fetch(`${BASE}/tasks/${taskId}/result`, {
    headers: HEADERS,
  });

  if (!resultResponse.ok) {
    throw new Error(`result failed: ${resultResponse.status}`);
  }

  const resultData = await resultResponse.json();
  return resultData.data.markdown;
}

convert("document.pdf")
  .then((markdown) => console.log(markdown.slice(0, 500)))
  .catch((error) => console.error(error));
```

### 9.4 浏览器端 SSE 监听

```javascript
const eventSource = new EventSource(
  "http://127.0.0.1:23504/api/v1/tasks/TASK_ID/progress?api_key=your-key"
);

eventSource.onmessage = (event) => {
  console.log("log:", event.data);
};

eventSource.addEventListener("done", (event) => {
  const payload = JSON.parse(event.data);
  console.log("done:", payload.elapsed);
  eventSource.close();
});

eventSource.addEventListener("error", (event) => {
  console.error("error:", event.data);
  eventSource.close();
});
```

说明：浏览器原生 EventSource 不支持自定义请求头，如果启用了 API Key 且你在浏览器里直接连 SSE，建议改用 query 参数传 api_key。

## 10. 集成建议

- 上传前先调用 /models，避免写死模型名。
- 下载型调用方建议优先走 /download，能一次拿到 Markdown 和图片。
- 只需要文本时优先走 /result/markdown，减少返回体体积。
- 浏览器端接 SSE 时优先用 api_key 查询参数，因为 EventSource 不能稳定附带自定义请求头。
- 任务状态拿到 error 时，直接读取 error 字段即可定位失败原因。

## 11. 备注

- 默认 Web 服务端口为 23504，可通过 WEB_PORT 环境变量修改。
- 服务默认最大上传体积为 200 MB。
- 图片输出目录固定为任务结果目录下的 images 子目录。