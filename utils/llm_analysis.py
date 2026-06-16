"""Optional LLM analysis for bot-detection predictions."""

import json
import time
from typing import Any

from .account_summary import (
    build_account_metrics,
    build_behavior_metrics,
    build_prompt_context,
)
from .llm_config import LLMConfig


BEHAVIOR_ANALYSIS_KEYS = (
    "like_behavior",
    "posting_behavior",
    "follow_behavior",
    "profile_behavior",
    "comment_behavior",
)

OUTPUT_FIELD_NAME_REPLACEMENTS = {
    "favourites_count": "点赞数",
    "statuses_count": "发帖数",
    "followers_count": "粉丝数",
    "friends_count": "关注数",
    "listed_count": "列表数",
    "created_at": "注册时间",
    "default_profile_image": "默认头像",
    "default_profile": "默认主页",
    "screen_name": "用户名",
    "description": "简介",
    "location": "位置",
    "url": "链接",
    "verified": "认证状态",
    "protected": "保护状态",
    "predicted_label": "预测标签",
    "comment_behavior": "评论行为",
    "confidence": "置信度",
    "probabilities": "概率",
}


def _sanitize_output_text(value: str) -> str:
    sanitized = value
    for raw, label in OUTPUT_FIELD_NAME_REPLACEMENTS.items():
        sanitized = sanitized.replace(raw, label)
    return sanitized


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            _sanitize_output_text(str(item).strip())
            for item in value
            if str(item).strip()
        ]
    if isinstance(value, str) and value.strip():
        return [_sanitize_output_text(value.strip())]
    return []


def _empty_behavior_anomalies() -> dict[str, list[str]]:
    return {key: [] for key in BEHAVIOR_ANALYSIS_KEYS}


def _as_behavior_anomalies(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return _empty_behavior_anomalies()
    return {key: _as_list(value.get(key)) for key in BEHAVIOR_ANALYSIS_KEYS}


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


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _format_percent(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "未知"
    percent = parsed * 100 if 0 <= parsed <= 1 else parsed
    return f"{percent:.2f}%"


def _prediction_explanation(prediction: dict) -> str:
    label = str(prediction.get("label", "未知")).strip() or "未知"
    confidence = _format_percent(prediction.get("confidence"))
    probabilities = prediction.get("probabilities", {})
    human = "未知"
    bot = "未知"
    if isinstance(probabilities, dict):
        human = _format_percent(probabilities.get("human"))
        bot = _format_percent(probabilities.get("bot"))

    return (
        f"模型预测结果为 {label}，置信度 {confidence}；"
        f"human 概率 {human}，bot 概率 {bot}。"
        "后续解读需围绕该预测结果展开。"
    )


def _base_analysis(status: str, user: dict, prediction: dict, summary: str) -> dict:
    return {
        "status": status,
        "risk_level": _risk_from_prediction(prediction),
        "prediction_explanation": _prediction_explanation(prediction),
        "summary": summary,
        "account_metrics": build_account_metrics(user),
        "behavior_metrics": build_behavior_metrics(user, prediction=prediction),
        "behavior_anomalies": _empty_behavior_anomalies(),
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
        return f"""请基于以下机器人账号检测结果和账号行为数据，生成中文结构化异常行为分析。

输入数据：
{context_json}

要求：
1. 只依据输入数据解释，不引用或假设外部事实。
2. 不输出思维链、隐藏推理或内部分析过程。
3. AI 分析必须符合 prediction.label、confidence、probabilities 的模型预测结果；summary 开头必须先用一句话解释模型预测标签、置信度和 human/bot 概率，再概括行为风险。
4. 不得推翻或弱化模型预测标签；如果行为证据较弱，也只能说明“需结合模型预测复核”，不能改判为相反标签。
5. 分析重点必须放在账号异常行为上，而不是泛泛复述账号资料。
6. 必须逐项分析以下维度，并在每条结论中引用接口输入值作为依据；不要引用或创造未在输入中出现的派生指标。
   - like_behavior：点赞行为，重点看点赞数。
   - posting_behavior：发帖行为，重点看发帖数和推文样本是否重复、广告化、诱导关注或链接导流。
   - follow_behavior：关注行为，重点看关注数、粉丝数、列表数。
   - profile_behavior：资料行为，重点看注册时间、认证状态、默认头像/主页、简介、位置和链接完整度。
   - comment_behavior：评论行为；没有评论数、回复数或评论内容字段，只能根据注册时间、发帖数、实际预测标签和可见推文样本间接解析评论行为倾向。
7. 评论行为分析禁止声称掌握评论数、回复数、评论内容、评论频率等未输入证据；必须明确这是基于注册时间、发帖数、实际预测标签和文本样本的间接判断。
8. key_factors 应优先列出由行为数据支持的异常点；content_signals 只描述推文样本中可见的内容信号。
9. 输出的中文文本中禁止出现输入 JSON 的英文字段名；请使用“点赞数”“发帖数”“粉丝数”“关注数”“列表数”“注册时间”“预测标签”等中文名称。
10. 输出必须是 JSON 对象，且只包含以下字段：
{{
  "risk_level": "low|medium|high",
  "summary": "先解释模型预测结果，再一句话概括账号异常行为风险",
  "behavior_anomalies": {{
    "like_behavior": ["点赞行为异常或正常的证据"],
    "posting_behavior": ["发帖行为异常或正常的证据"],
    "follow_behavior": ["关注行为异常或正常的证据"],
    "profile_behavior": ["资料行为异常或正常的证据"],
    "comment_behavior": ["评论行为间接解析的证据"]
  }},
  "key_factors": ["影响判断的行为特征"],
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
        summary = _sanitize_output_text(str(parsed.get("summary", "")).strip())
        if not summary:
            raise ValueError("LLM JSON 缺少 summary")

        return {
            "status": "success",
            "risk_level": _risk_from_prediction(prediction),
            "prediction_explanation": _prediction_explanation(prediction),
            "summary": summary,
            "account_metrics": build_account_metrics(user),
            "behavior_metrics": build_behavior_metrics(user, prediction=prediction),
            "behavior_anomalies": _as_behavior_anomalies(
                parsed.get("behavior_anomalies")
            ),
            "key_factors": _as_list(parsed.get("key_factors")),
            "content_signals": _as_list(parsed.get("content_signals")),
            "recommendations": _as_list(parsed.get("recommendations")),
        }
