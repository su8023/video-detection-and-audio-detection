# Video Detection and Audio Detection

本项目是一个本地优先的桌面监控测试程序，使用 Python Flask 后端和 HTML/CSS/JavaScript 前端，实现摄像头实时画面、人员检测、麦克风实时转写、历史记录、阶段总结和本地 ASR 模型切换。

代码可以上传到 GitHub 或打包分享；模型、数据库、虚拟环境不随仓库提交，部署时再下载。

## 功能

- 摄像头接入和 MJPEG 实时画面回传
- YOLOv8 人员检测，支持 CPU/CUDA
- 麦克风接入、录音测试和实时语音预览
- Vosk 低延迟流式识别
- Qwen3-ASR / Whisper 最终校正
- 转写断句后立即显示卡片，模型校正完成后原地更新
- SQLite 保存转写记录、人员事件和阶段总结
- 历史记录搜索、清空和 MD/TXT/JSON 导出
- 人员离开监控范围后自动生成阶段总结
- 设置页支持切换语音延迟模式、ASR 模型和自动总结参数

## 目录结构

```text
app.py                         Flask 后端入口、API、摄像头服务、自动化逻辑
config.yaml                    主配置文件
requirements-core.txt          核心依赖
requirements-ai.txt            AI/模型相关依赖
templates/index.html           前端页面
static/js/app.js               前端交互逻辑
static/css/style.css           前端样式
core/stt_worker.py             实时语音转写和 ASR 校正
core/person_detector.py        YOLO 人员检测
core/database.py               SQLite 数据库读写
core/summarizer.py             阶段总结
scripts/hardware_check.py      本机硬件/环境检测
scripts/mic_stt_test.py        麦克风/语音识别测试
scripts/prepare_models.py      部署时下载模型
scripts/make_package.ps1       打包代码，不包含模型和数据
```

运行时会生成或下载以下内容，默认不提交到仓库：

```text
.venv/                         Python 虚拟环境
models/                        YOLO、Vosk 等本地模型
data/app.db                    SQLite 历史数据库
__pycache__/                   Python 缓存
```

## 新电脑部署

Windows PowerShell 示例：

```powershell
cd D:\local-ai-monitor
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-core.txt
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
.\.venv\Scripts\python scripts\prepare_models.py
```

`prepare_models.py` 默认下载：

```text
models/yolov8n.pt
models/vosk-model-small-cn-0.22
```

如需提前下载 ASR 模型：

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

不提前下载也可以启动服务。首次使用 Qwen3-ASR 或 Whisper 时，程序会自动下载到 Hugging Face 缓存。

## 启动

```powershell
cd D:\local-ai-monitor
.\.venv\Scripts\python app.py
```

浏览器打开：

```text
http://127.0.0.1:5050/
```

后台启动：

```powershell
Start-Process -FilePath "D:\local-ai-monitor\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory "D:\local-ai-monitor" -WindowStyle Hidden
```

## 停止

```powershell
$pids = Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
$pids | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```

确认端口是否释放：

```powershell
$conn = Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue
if ($conn) { $conn } else { "NO_LISTENER" }
```

## 打包分享

推荐使用内置脚本：

```powershell
cd D:\local-ai-monitor
powershell -ExecutionPolicy Bypass -File .\scripts\make_package.ps1 -Output D:\local-ai-monitor-package.zip
```

压缩包只包含代码、配置、前端、脚本和依赖清单，不包含：

```text
.venv/
models/
data/
__pycache__/
*.zip
```

这样别人拿到代码后，在自己的电脑上安装依赖并下载模型即可运行，不需要传输大模型文件。

## 配置

主配置文件：

```text
config.yaml
```

常用配置：

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

## ASR 模型

默认最终校正模型：

```text
Qwen/Qwen3-ASR-0.6B
```

页面里可以切换：

```text
Qwen3-ASR 0.6B       推荐默认测试，中英混合表现较好，显存压力较低
Qwen3-ASR 1.7B       准确率更强，但加载更慢、显存占用更高
Whisper medium       稳定备用
Whisper small        更轻量，低延迟备用
```

第一次加载大模型会比较慢，下载和显存加载完成后再评估实时表现。

## 页面使用

1. 打开 `http://127.0.0.1:5050/`
2. 查看左侧摄像头画面和人员检测框
3. 点击“开始转写”
4. 说话后会先出现实时卡片并显示“校正中”
5. 模型完成校正后，卡片会原地更新为最终文本
6. 历史记录区可以搜索、清空和导出
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
- 摄像头和麦克风会占用硬件设备，无法打开时先检查是否被其他软件占用。
- 切换 ASR 模型时，如果正在转写，当前转写会停止，需要重新点击“开始转写”。
- `data/app.db` 是本地数据库文件，删除后历史记录、总结和事件会丢失。
- `.venv` 是本机虚拟环境，不建议提交或打包。
