"""Optional LLM analysis for bot-detection predictions."""

import json
import time
from typing import Any

from .account_summary import build_account_metrics, build_prompt_context
from .llm_config import LLMConfig


VALID_RISK_LEVELS = {"low", "medium", "high"}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _risk_from_prediction(prediction: dict) -> str:
    probabilities = prediction.get("probabilities", {})
    bot_probability = 0.0
    if isinstance(probabilities, dict):
        try:
            bot_probability = float(probabilities.get("bot", 0.0))
        except (TypeError, ValueError):
            bot_probability = 0.0

    if bot_probability >= 0.75:
        return "high"
    if bot_probability >= 0.45:
        return "medium"
    return "low"


def _base_analysis(status: str, user: dict, prediction: dict, summary: str) -> dict:
    return {
        "status": status,
        "risk_level": _risk_from_prediction(prediction),
        "summary": summary,
        "account_metrics": build_account_metrics(user),
        "key_factors": [],
        "content_signals": [],
        "recommendations": [],
    }


def skipped_analysis(user: dict, prediction: dict, reason: str) -> dict:
    analysis = _base_analysis("skipped", user, prediction, reason)
    analysis["recommendations"] = ["如需自动解读，请启用 ENABLE_LLM 并配置 OPENAI_API_KEY"]
    return analysis


def error_analysis(user: dict, prediction: dict, error: str) -> dict:
    analysis = _base_analysis("error", user, prediction, f"LLM 解读失败：{error}")
    analysis["recommendations"] = ["请查看服务日志或稍后重试；原始模型预测仍可作为参考"]
    return analysis


class LLMAnalyzer:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

    def public_status(self) -> dict:
        return self.config.public_status()

    def analyze_prediction(self, user: dict, prediction: dict) -> dict:
        if not self.config.enabled:
            return skipped_analysis(user, prediction, "LLM 解读未启用")
        if not self.config.api_key.strip():
            return skipped_analysis(user, prediction, "LLM 解读未配置 OPENAI_API_KEY")

        try:
            prompt = self._build_prompt(user, prediction)
            content = self._call_chat_completions(prompt)
            parsed = self._parse_response(content)
            return self._normalize_result(parsed, user, prediction)
        except Exception as exc:
            return error_analysis(user, prediction, str(exc))

    def _build_prompt(self, user: dict, prediction: dict) -> str:
        context = build_prompt_context(user, prediction)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)
        return f"""请基于以下机器人账号检测结果和账号资料，生成中文结构化解读。

输入数据：
{context_json}

要求：
1. 只依据输入数据解释，不引用或假设外部事实。
2. 不输出思维链、隐藏推理或内部分析过程。
3. 重点解释预测标签、置信度、点赞数、发帖数、粉丝数、关注数、粉关比、账号年龄、认证状态、默认头像/主页、资料完整度和推文内容信号。
4. 输出必须是 JSON 对象，且只包含以下字段：
{{
  "risk_level": "low|medium|high",
  "summary": "一句话结论",
  "key_factors": ["影响判断的账号特征"],
  "content_signals": ["推文内容层面的信号"],
  "recommendations": ["处置或复核建议"]
}}"""

    def _call_chat_completions(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests 依赖未安装，无法调用 LLM API") from exc

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是社交媒体账号风控分析助手，只返回可解析的 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }

        attempts = max(1, self.config.max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("LLM 返回内容为空")
                    return content.strip()

                last_error = RuntimeError(
                    f"LLM API 请求失败: {response.status_code} - {response.text}"
                )
            except requests.RequestException as exc:
                last_error = exc

            if attempt < attempts - 1:
                time.sleep(2 ** attempt)

        if last_error is None:
            raise RuntimeError("LLM API 请求失败")
        raise last_error

    def _parse_response(self, content: str) -> dict:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("LLM 返回不是 JSON")
            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM JSON 解析失败: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON 顶层必须是对象")
        return parsed

    def _normalize_result(self, parsed: dict, user: dict, prediction: dict) -> dict:
        risk_level = str(parsed.get("risk_level", "")).strip().lower()
        if risk_level not in VALID_RISK_LEVELS:
            risk_level = _risk_from_prediction(prediction)

        summary = str(parsed.get("summary", "")).strip()
        if not summary:
            raise ValueError("LLM JSON 缺少 summary")

        return {
            "status": "success",
            "risk_level": risk_level,
            "summary": summary,
            "account_metrics": build_account_metrics(user),
            "key_factors": _as_list(parsed.get("key_factors")),
            "content_signals": _as_list(parsed.get("content_signals")),
            "recommendations": _as_list(parsed.get("recommendations")),
        }
