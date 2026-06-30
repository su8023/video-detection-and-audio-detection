# Local AI Monitor

本项目是一个本地优先的桌面监控测试程序，使用 Python Flask 后端和 HTML/CSS/JavaScript 前端，实现摄像头实时画面、人员检测、麦克风语音转写、历史记录、阶段总结和本地 ASR 模型切换。

## 项目位置

全部代码和配置文件都在：

```text
D:\local-ai-monitor
```

主要代码文件：

```text
D:\local-ai-monitor\app.py                      Flask 后端入口、API、摄像头服务、自动化逻辑
D:\local-ai-monitor\config.yaml                 主配置文件
D:\local-ai-monitor\templates\index.html        前端页面
D:\local-ai-monitor\static\js\app.js            前端交互逻辑
D:\local-ai-monitor\static\css\style.css        前端样式
D:\local-ai-monitor\core\stt_worker.py          实时语音转写和 ASR 校正逻辑
D:\local-ai-monitor\core\person_detector.py     YOLO 人员检测逻辑
D:\local-ai-monitor\core\database.py            SQLite 数据库读写
D:\local-ai-monitor\core\summarizer.py          阶段总结逻辑
D:\local-ai-monitor\scripts\hardware_check.py   本机硬件/环境检测脚本
D:\local-ai-monitor\scripts\mic_stt_test.py     麦克风/语音识别测试脚本
```

数据和模型目录：

```text
D:\local-ai-monitor\data\app.db                 SQLite 数据库，保存历史记录/总结/事件
D:\local-ai-monitor\models\                    本地模型目录，如 YOLO、Vosk 模型
D:\local-ai-monitor\.venv\                     Python 虚拟环境
```

## 当前功能

- 摄像头接入和实时画面回传
- YOLOv8 人员检测，支持 CUDA
- 麦克风接入和音量测试
- Vosk 低延迟实时语音预览
- ASR 最终校正模型可切换：
  - Qwen3-ASR 0.6B
  - Qwen3-ASR 1.7B
  - OpenAI Whisper medium
  - OpenAI Whisper small
- 转写卡片实时显示：断句后先显示“校正中”，模型完成后原地更新
- 语音历史记录保存到 SQLite
- 历史记录搜索和导出：Markdown、TXT、JSON
- 人员进入/离开事件记录
- 人员离开画面后自动生成阶段总结
- 阶段总结列表过滤空内容
- 设置页支持切换语音延迟模式、ASR 模型和自动总结参数

## 启动服务

打开 PowerShell：

```powershell
cd D:\local-ai-monitor
.\.venv\Scripts\python app.py
```

然后在浏览器打开：

```text
http://127.0.0.1:5050/
```

如果想后台启动：

```powershell
Start-Process -FilePath "D:\local-ai-monitor\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory "D:\local-ai-monitor" -WindowStyle Hidden
```

## 停止服务

按端口停止：

```powershell
$pids = Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
$pids | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

确认端口是否释放：

```powershell
$conn = Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue
if ($conn) { $conn } else { "NO_LISTENER" }
```

## 安装依赖

如果虚拟环境已经存在，一般不需要重新安装。

核心依赖：

```powershell
cd D:\local-ai-monitor
.\.venv\Scripts\python -m pip install -r requirements-core.txt
```

AI 功能依赖：

```powershell
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
```

## 分发给别人

不要把下面这些目录一起压缩发送：

```text
D:\local-ai-monitor\.venv
D:\local-ai-monitor\data
D:\local-ai-monitor\models
D:\local-ai-monitor\__pycache__
```

原因：

- `.venv` 是本机虚拟环境，体积大，而且换电脑不一定可用。
- `data` 是本机历史记录数据库。
- `models` 是运行时模型文件，可以部署时再下载。
- Qwen/Whisper 这类模型会进入 Hugging Face 缓存，也不建议打包进项目。

推荐用内置打包脚本：

```powershell
cd D:\local-ai-monitor
.\scripts\make_package.ps1 -Output D:\local-ai-monitor-package.zip
```

如果 Windows 提示脚本执行策略禁止运行，可以用：

```powershell
powershell -ExecutionPolicy Bypass -File D:\local-ai-monitor\scripts\make_package.ps1 -Output D:\local-ai-monitor-package.zip
```

这个压缩包只包含代码、配置、前端、脚本和依赖清单，不包含大模型、数据库和虚拟环境。

## 新电脑部署流程

别人拿到压缩包后：

1. 解压到目标目录，例如：

```text
D:\local-ai-monitor
```

2. 创建虚拟环境：

```powershell
cd D:\local-ai-monitor
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

3. 安装依赖：

```powershell
.\.venv\Scripts\python -m pip install -r requirements-core.txt
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
```

4. 下载运行必需的小模型：

```powershell
.\.venv\Scripts\python scripts\prepare_models.py
```

默认会下载：

```text
models\yolov8n.pt
models\vosk-model-small-cn-0.22
```

5. 如果想提前下载 ASR 大模型，可以运行：

```powershell
.\.venv\Scripts\python scripts\prepare_models.py --skip-core --asr qwen3_0_6b
```

可选值：

```text
qwen3_0_6b
qwen3_1_7b
whisper_medium
whisper_small
all
```

如果不提前下载 Qwen/Whisper，也可以直接启动服务，第一次点击“开始转写”或第一次加载模型时会自动下载到 Hugging Face 缓存。

## 配置文件

主配置文件：

```text
D:\local-ai-monitor\config.yaml
```

常用配置项：

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

vision:
  enabled: true
  model_path: models/yolov8n.pt
  device: cuda

audio:
  sample_rate: 16000
  input_device: null

stt:
  engine: vosk_stream
  finalizer_backend: qwen3_asr
  finalizer_model_id: Qwen/Qwen3-ASR-0.6B
  finalizer_device: cuda

automation:
  enabled: true
  person_absent_summary: true
  absent_confirm_seconds: 5
```

## ASR 模型说明

当前默认最终校正模型：

```text
Qwen/Qwen3-ASR-0.6B
```

页面中可以切换：

```text
Qwen3-ASR 0.6B       中文/中英混合优先，先推荐测试这个
Qwen3-ASR 1.7B       准确率更高，但显存和加载时间更高
Whisper medium       稳定备用
Whisper small        更轻量，低延迟备用
```

第一次启动 Qwen 或 Whisper 模型时，可能会下载模型并加载到显存，等待时间会比较长。模型加载完成后再测试准确率和延迟。

## 页面使用

1. 打开 `http://127.0.0.1:5050/`
2. 查看左侧摄像头画面和人员检测框
3. 在右侧点击“开始转写”
4. 说话后，实时转写区会先出现卡片并显示“校正中”
5. ASR 模型校正完成后，卡片会原地更新为最终文本
6. 历史记录区可以搜索、刷新、清空和导出
7. 阶段总结区可以手动生成总结
8. 设置区可以切换语音模式、ASR 模型和自动总结参数

## 硬件检测

```powershell
cd D:\local-ai-monitor
.\.venv\Scripts\python scripts\hardware_check.py
```

## 常用接口

```text
GET  /api/status                 查看摄像头、语音、数据库、工具状态
POST /api/mic-test               麦克风录音测试
POST /api/stt/start              开始实时转写
POST /api/stt/stop               停止实时转写
POST /api/stt/clear              清空当前实时转写列表
GET  /api/transcripts            当前实时转写状态和记录
GET  /api/history                历史转写记录
GET  /api/history/search         搜索历史记录
GET  /api/history/export         导出历史记录
GET  /api/summaries              阶段总结列表
POST /api/summary/generate       手动生成总结
GET  /api/events                 事件列表
GET  /api/settings               设置项
POST /api/settings/voice-mode    切换语音延迟模式
POST /api/settings/asr-model     切换 ASR 模型
POST /api/settings/automation    修改自动化设置
```

## 注意事项

- 本项目优先本地运行，不依赖云端 API。
- 摄像头和麦克风会占用硬件设备，如果无法打开，先检查是否被其他软件占用。
- Qwen3-ASR 和 Whisper 首次加载会占用显存，切换模型时如果正在转写，服务会停止当前转写，需要重新点击“开始转写”。
- `data\app.db` 是本地数据库文件，删除后历史记录、总结和事件会丢失。
- `.venv` 是虚拟环境目录，通常不要手动修改。
