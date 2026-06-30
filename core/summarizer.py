from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any


class LocalSummarizer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.provider = str(cfg.get("provider", "ollama"))
        self.model = str(cfg.get("model", "qwen2.5:7b-instruct-q4_K_M"))
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:11434"))
        self.timeout = int(cfg.get("timeout_seconds", 90))

    def summarize(self, transcripts: list[dict[str, Any]]) -> dict[str, Any]:
        if not transcripts:
            return {
                "title": "暂无可总结内容",
                "summary": "当前没有历史转写记录。",
                "provider": "local-rule",
                "model": "fallback",
            }

        prompt = self._build_prompt(transcripts)
        if self.provider == "ollama":
            try:
                text = self._summarize_with_ollama(prompt)
                return {
                    "title": self._make_title(text),
                    "summary": text,
                    "provider": "ollama",
                    "model": self.model,
                }
            except Exception as exc:
                fallback = self._fallback_summary(transcripts)
                fallback["summary"] += f"\n\n本地模型暂不可用，已使用规则摘要。错误：{exc}"
                return fallback

        return self._fallback_summary(transcripts)

    def _summarize_with_ollama(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = str(result.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text

    def _build_prompt(self, transcripts: list[dict[str, Any]]) -> str:
        lines = []
        for item in transcripts:
            speaker = item.get("speaker") or "说话人"
            time_text = item.get("display_time") or item.get("created_at") or ""
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"[{time_text}] {speaker}: {text}")

        content = "\n".join(lines[-120:])
        return (
            "你是一个本地会议/监控记录助手。请基于以下转写记录生成中文总结。\n"
            "要求：\n"
            "1. 提炼发生了什么。\n"
            "2. 按说话人或主题归纳重点。\n"
            "3. 列出待办、问题或风险。\n"
            "4. 如果记录很碎或可能有识别错误，请明确说明不确定之处。\n\n"
            f"转写记录：\n{content}\n\n"
            "请输出：\n"
            "## 概要\n"
            "## 重点\n"
            "## 待办/问题\n"
            "## 不确定信息\n"
        )

    def _fallback_summary(self, transcripts: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item.get("text") or "").strip() for item in transcripts if item.get("text")]
        joined = " ".join(texts)
        keywords = self._keywords(joined)
        preview = "\n".join(f"- {text}" for text in texts[-8:])
        summary = (
            "## 概要\n"
            f"共整理 {len(texts)} 条转写记录。本地大模型未启用时，当前为规则摘要。\n\n"
            "## 关键词\n"
            f"{'、'.join(keywords) if keywords else '暂无明显关键词'}\n\n"
            "## 最近内容\n"
            f"{preview or '- 暂无内容'}\n\n"
            "## 待办/问题\n"
            "- 建议安装或启动 Ollama 后生成更高质量总结。\n"
        )
        return {
            "title": "规则摘要",
            "summary": summary,
            "provider": "local-rule",
            "model": "fallback",
        }

    def _keywords(self, text: str) -> list[str]:
        words = re.findall(r"[\u4e00-\u9fff]{2,6}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
        ignore = {"这个", "那个", "然后", "就是", "可以", "现在", "没有", "还是", "因为"}
        counts: dict[str, int] = {}
        for word in words:
            if word in ignore:
                continue
            counts[word] = counts.get(word, 0) + 1
        return [word for word, _count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]]

    def _make_title(self, text: str) -> str:
        for line in text.splitlines():
            clean = line.strip("# -*")
            if clean:
                return clean[:30]
        return "阶段总结"
