const fields = {
  cameraState: document.querySelector("#cameraState"),
  frameSize: document.querySelector("#frameSize"),
  captureFps: document.querySelector("#captureFps"),
  peopleCount: document.querySelector("#peopleCount"),
  detectorState: document.querySelector("#detectorState"),
  inferenceTime: document.querySelector("#inferenceTime"),
  databaseState: document.querySelector("#databaseState"),
  automationState: document.querySelector("#automationState"),
  ollamaState: document.querySelector("#ollamaState"),
  deviceList: document.querySelector("#deviceList"),
  micResult: document.querySelector("#micResult"),
  sttStatus: document.querySelector("#sttStatus"),
  transcriptList: document.querySelector("#transcriptList"),
  historyList: document.querySelector("#historyList"),
  historySearchInput: document.querySelector("#historySearchInput"),
  summaryStatus: document.querySelector("#summaryStatus"),
  summaryList: document.querySelector("#summaryList"),
  eventList: document.querySelector("#eventList"),
  voiceModeSelect: document.querySelector("#voiceModeSelect"),
  voiceModeHint: document.querySelector("#voiceModeHint"),
  asrModelSelect: document.querySelector("#asrModelSelect"),
  asrModelHint: document.querySelector("#asrModelHint"),
  autoSummaryEnabled: document.querySelector("#autoSummaryEnabled"),
  absentSecondsInput: document.querySelector("#absentSecondsInput"),
  settingsStatus: document.querySelector("#settingsStatus"),
};

function yesNo(value) {
  return value ? "正常" : "未就绪";
}

function formatMs(value) {
  return typeof value === "number" ? `${value.toFixed(0)} ms` : "-";
}

function renderSttStatus(status) {
  if (!status) return "未启动";
  const engine = status.engine || "unknown";
  const decode = formatMs(status.last_decode_ms);
  const endpoint = formatMs(status.last_endpoint_delay_ms);
  const finalizer = status.finalizer_enabled
    ? ` | 校正 ${status.finalizer_loading ? "加载中" : formatMs(status.finalizer_last_ms)} | 队列 ${
        status.finalizer_pending || 0
      }`
    : "";
  if (status.loading) return `正在加载 ${engine} 模型...`;
  if (status.running) {
    return `运行中 | ${engine} | 断句 ${endpoint} | 解码 ${decode}${finalizer} | RMS ${Number(
      status.last_rms || 0
    ).toFixed(5)} | ${status.records_count || 0} 段`;
  }
  if (status.last_error) return `错误：${status.last_error}`;
  return "未启动";
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  const camera = data.camera || {};
  const detector = camera.detector || {};

  fields.cameraState.textContent = camera.opened ? "已打开" : camera.last_error || "未打开";
  fields.frameSize.textContent =
    camera.frame_width && camera.frame_height ? `${camera.frame_width} x ${camera.frame_height}` : "-";
  fields.captureFps.textContent = camera.capture_fps ? `${camera.capture_fps}` : "-";
  fields.peopleCount.textContent = `${camera.people_count ?? 0}`;
  fields.detectorState.textContent = detector.enabled ? `已启用 ${detector.imgsz}` : detector.error || "未启用";
  fields.inferenceTime.textContent = formatMs(detector.last_inference_ms);
  fields.databaseState.textContent = data.database ? `${data.database.transcripts} 条` : "-";
  fields.automationState.textContent = data.automation?.enabled
    ? data.automation.last_event || "运行中"
    : "未启用";
  fields.ollamaState.textContent = yesNo(data.tools?.ollama);
  fields.sttStatus.textContent = renderSttStatus(data.stt);

  fields.deviceList.innerHTML = "";
  for (const dev of data.audio_input_devices || []) {
    const li = document.createElement("li");
    li.textContent = `#${dev.index} ${dev.name} | 输入通道 ${dev.max_input_channels} | ${dev.default_samplerate} Hz`;
    fields.deviceList.appendChild(li);
  }
}

function transcriptItem(rec, className = "transcript-item") {
  const item = document.createElement("article");
  item.className = className;
  if (rec.pending) item.classList.add("transcript-pending");

  const meta = document.createElement("div");
  meta.className = "transcript-meta";
  const speaker = rec.speaker || "说话人";
  const time = rec.time || rec.display_time || rec.created_at || "";
  const engine = rec.engine ? ` | ${rec.engine}` : "";
  const language = rec.language ? ` | ${rec.language}` : "";
  const pending = rec.pending ? " | 校正中" : "";
  meta.textContent = `${time} | ${speaker}${engine}${language}${pending} | ${formatMs(rec.elapsed_ms)}`;

  const text = document.createElement("p");
  text.textContent = rec.text || "正在识别...";

  item.appendChild(meta);
  item.appendChild(text);

  if (rec.draft_text && rec.draft_text !== rec.text) {
    const draft = document.createElement("div");
    draft.className = "transcript-draft";
    draft.textContent = `实时草稿：${rec.draft_text}`;
    item.appendChild(draft);
  }

  return item;
}

function renderPartial(status) {
  const partial = status?.partial || "";
  if (!partial) return null;
  const item = document.createElement("article");
  item.className = "transcript-item transcript-partial";

  const meta = document.createElement("div");
  meta.className = "transcript-meta";
  meta.textContent = `实时预览 | 延迟 ${status.partial_age_ms ?? 0} ms`;

  const text = document.createElement("p");
  text.textContent = partial;

  item.appendChild(meta);
  item.appendChild(text);
  return item;
}

async function refreshTranscripts() {
  const res = await fetch("/api/transcripts");
  const data = await res.json();
  const status = data.status || {};
  fields.sttStatus.textContent = renderSttStatus(status);
  const records = data.records || [];
  fields.transcriptList.innerHTML = "";

  const partialItem = renderPartial(status);
  if (partialItem) fields.transcriptList.appendChild(partialItem);

  if (records.length === 0 && !partialItem) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "暂无转写内容";
    fields.transcriptList.appendChild(empty);
    return;
  }

  for (const rec of records.slice().reverse()) {
    fields.transcriptList.appendChild(transcriptItem(rec));
  }
}

let currentHistoryQuery = "";

async function refreshHistory() {
  const q = currentHistoryQuery.trim();
  const url = q ? `/api/history/search?limit=80&q=${encodeURIComponent(q)}` : "/api/history?limit=80";
  const res = await fetch(url);
  const data = await res.json();
  const records = data.records || [];
  fields.historyList.innerHTML = "";
  if (records.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "暂无历史记录";
    fields.historyList.appendChild(empty);
    return;
  }
  for (const rec of records) {
    fields.historyList.appendChild(transcriptItem(rec, "transcript-item history-item"));
  }
}

async function refreshSettings() {
  const res = await fetch("/api/settings");
  const data = await res.json();
  if (fields.voiceModeSelect) {
    fields.voiceModeSelect.value = data.voice_mode === "custom" ? "balanced" : data.voice_mode;
  }
  const mode = fields.voiceModeSelect?.value || data.voice_mode;
  const modeInfo = data.voice_modes?.[mode];
  fields.voiceModeHint.textContent = modeInfo ? modeInfo.description : "自定义配置";
  if (fields.asrModelSelect) {
    fields.asrModelSelect.value = data.finalizer_model === "custom" ? "qwen3_0_6b" : data.finalizer_model;
  }
  const model = fields.asrModelSelect?.value || data.finalizer_model;
  const modelInfo = data.finalizer_models?.[model];
  fields.asrModelHint.textContent = modelInfo ? modelInfo.description : "自定义模型配置";
  fields.autoSummaryEnabled.checked = Boolean(data.automation?.person_absent_summary);
  fields.absentSecondsInput.value = data.automation?.absent_confirm_seconds ?? 5;
}

function exportHistory(format) {
  const q = currentHistoryQuery.trim();
  const url = `/api/history/export?format=${format}&limit=1000${q ? `&q=${encodeURIComponent(q)}` : ""}`;
  window.location.href = url;
}

async function refreshSummaries() {
  const res = await fetch("/api/summaries?limit=20");
  const data = await res.json();
  const summaries = (data.summaries || []).filter((item) => (item.summary || "").trim());
  fields.summaryList.innerHTML = "";
  if (summaries.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "暂无总结";
    fields.summaryList.appendChild(empty);
    return;
  }
  for (const item of summaries) {
    const article = document.createElement("article");
    article.className = "summary-item";

    const meta = document.createElement("div");
    meta.className = "transcript-meta";
    meta.textContent = `${item.created_at} | ${item.provider} | ${item.model} | ${item.transcript_count} 条`;

    const title = document.createElement("h3");
    title.textContent = item.title || "阶段总结";

    const body = document.createElement("pre");
    body.textContent = item.summary || "";

    article.appendChild(meta);
    article.appendChild(title);
    article.appendChild(body);
    fields.summaryList.appendChild(article);
  }
}

async function refreshEvents() {
  const res = await fetch("/api/events?limit=80");
  const data = await res.json();
  const events = data.events || [];
  fields.eventList.innerHTML = "";
  if (events.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "暂无事件";
    fields.eventList.appendChild(empty);
    return;
  }
  for (const event of events) {
    const item = document.createElement("article");
    item.className = "event-item";

    const meta = document.createElement("div");
    meta.className = "transcript-meta";
    meta.textContent = `${event.created_at} | ${event.event_type} | 人数 ${event.people_count ?? "-"}`;

    const title = document.createElement("p");
    title.textContent = event.title || "";

    item.appendChild(meta);
    item.appendChild(title);

    if (event.detail) {
      const detail = document.createElement("div");
      detail.className = "transcript-draft";
      detail.textContent = event.detail;
      item.appendChild(detail);
    }

    fields.eventList.appendChild(item);
  }
}

async function postJson(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function runMicTest() {
  fields.micResult.textContent = "正在录音 3 秒...";
  try {
    const data = await postJson("/api/mic-test", { seconds: 3 });
    fields.micResult.textContent = `RMS ${data.rms.toFixed(5)}，峰值 ${data.peak.toFixed(5)}`;
  } catch (err) {
    fields.micResult.textContent = `测试失败：${err}`;
  }
}

document.querySelector("#refreshBtn").addEventListener("click", async () => {
  await refreshStatus();
  await refreshHistory();
  await refreshSummaries();
  await refreshEvents();
});
document.querySelector("#micBtn").addEventListener("click", runMicTest);
document.querySelector("#sttStartBtn").addEventListener("click", async () => {
  fields.sttStatus.textContent = "正在启动...";
  await postJson("/api/stt/start");
  await refreshStatus();
});
document.querySelector("#sttStopBtn").addEventListener("click", async () => {
  await postJson("/api/stt/stop");
  await refreshStatus();
});
document.querySelector("#sttClearBtn").addEventListener("click", async () => {
  await postJson("/api/stt/clear");
  await refreshTranscripts();
});
document.querySelector("#historyRefreshBtn").addEventListener("click", refreshHistory);
document.querySelector("#historySearchBtn").addEventListener("click", async () => {
  currentHistoryQuery = fields.historySearchInput.value || "";
  await refreshHistory();
});
fields.historySearchInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter") {
    currentHistoryQuery = fields.historySearchInput.value || "";
    await refreshHistory();
  }
});
document.querySelector("#historyClearBtn").addEventListener("click", async () => {
  await postJson("/api/history/clear");
  await refreshHistory();
  await refreshStatus();
});
document.querySelector("#exportMdBtn").addEventListener("click", () => exportHistory("md"));
document.querySelector("#exportTxtBtn").addEventListener("click", () => exportHistory("txt"));
document.querySelector("#exportJsonBtn").addEventListener("click", () => exportHistory("json"));
document.querySelector("#saveVoiceModeBtn").addEventListener("click", async () => {
  const mode = fields.voiceModeSelect.value;
  fields.settingsStatus.textContent = "正在应用语音模式...";
  const result = await postJson("/api/settings/voice-mode", { mode });
  fields.settingsStatus.textContent = result.stopped_transcriber
    ? "已应用，转写已停止，请重新开始"
    : "已应用";
  await refreshSettings();
  await refreshStatus();
});
fields.voiceModeSelect.addEventListener("change", async () => {
  const res = await fetch("/api/settings");
  const data = await res.json();
  fields.voiceModeHint.textContent = data.voice_modes?.[fields.voiceModeSelect.value]?.description || "-";
});
document.querySelector("#saveAsrModelBtn").addEventListener("click", async () => {
  const model = fields.asrModelSelect.value;
  fields.settingsStatus.textContent = "正在切换 ASR 模型...";
  const result = await postJson("/api/settings/asr-model", { model });
  fields.settingsStatus.textContent = result.stopped_transcriber
    ? "模型已切换，转写已停止，请重新开始"
    : "ASR 模型已切换";
  await refreshSettings();
  await refreshStatus();
});
fields.asrModelSelect.addEventListener("change", async () => {
  const res = await fetch("/api/settings");
  const data = await res.json();
  fields.asrModelHint.textContent = data.finalizer_models?.[fields.asrModelSelect.value]?.description || "-";
});
document.querySelector("#saveAutomationBtn").addEventListener("click", async () => {
  fields.settingsStatus.textContent = "正在保存自动化设置...";
  await postJson("/api/settings/automation", {
    person_absent_summary: fields.autoSummaryEnabled.checked,
    absent_confirm_seconds: Number(fields.absentSecondsInput.value || 5),
  });
  fields.settingsStatus.textContent = "自动化设置已保存";
  await refreshSettings();
  await refreshStatus();
});
document.querySelector("#summaryRefreshBtn").addEventListener("click", refreshSummaries);
document.querySelector("#summaryBtn").addEventListener("click", async () => {
  fields.summaryStatus.textContent = "正在生成总结...";
  const result = await postJson("/api/summary/generate", { limit: 80 });
  fields.summaryStatus.textContent = `已生成：${result.provider} / ${result.model} / ${result.transcript_count} 条`;
  await refreshSummaries();
  await refreshStatus();
});
document.querySelector("#eventsRefreshBtn").addEventListener("click", refreshEvents);
document.querySelector("#eventsClearBtn").addEventListener("click", async () => {
  await postJson("/api/events/clear");
  await refreshEvents();
  await refreshStatus();
});

refreshStatus();
refreshTranscripts();
refreshHistory();
refreshSummaries();
refreshEvents();
refreshSettings();
setInterval(refreshStatus, 1000);
setInterval(refreshTranscripts, 100);
setInterval(refreshHistory, 5000);
setInterval(refreshEvents, 5000);
