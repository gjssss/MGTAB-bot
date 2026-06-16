"""Account-level metrics used by deterministic and LLM explanations."""

from datetime import datetime, timezone
from typing import Any


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _raw_str(user: dict, key: str) -> str:
    value = user.get(key, "")
    return value if isinstance(value, str) else ""


def _parse_created_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None

    raw = value.strip()
    formats = (
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a %b %d %H:%M:%S %z %Y",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def account_age_days(user: dict, now: datetime | None = None) -> int:
    created = _parse_created_at(user.get("created_at"))
    if created is None:
        return 0

    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    delta = resolved_now.astimezone(timezone.utc) - created
    return max(0, delta.days)


def build_account_metrics(user: dict, now: datetime | None = None) -> dict:
    followers = _to_int(user.get("followers_count"))
    friends = _to_int(user.get("friends_count"))
    tweets = user.get("tweets", [])
    if not isinstance(tweets, list):
        tweets = []

    return {
        "followers_count": followers,
        "friends_count": friends,
        "listed_count": _to_int(user.get("listed_count")),
        "favourites_count": _to_int(user.get("favourites_count")),
        "statuses_count": _to_int(user.get("statuses_count")),
        "followers_friends_ratio": round(followers / (friends + 1), 4),
        "tweet_count": len([item for item in tweets if isinstance(item, str) and item.strip()]),
        "account_age_days": account_age_days(user, now=now),
        "verified": bool(user.get("verified", False)),
        "protected": bool(user.get("protected", False)),
        "default_profile_image": bool(user.get("default_profile_image", False)),
        "default_profile": bool(user.get("default_profile", False)),
    }


def build_behavior_metrics(
    user: dict,
    now: datetime | None = None,
    prediction: dict | None = None,
) -> dict:
    followers = _to_int(user.get("followers_count"))
    friends = _to_int(user.get("friends_count"))
    favourites = _to_int(user.get("favourites_count"))
    statuses = _to_int(user.get("statuses_count"))
    listed = _to_int(user.get("listed_count"))
    predicted_label = prediction.get("label") if isinstance(prediction, dict) else None

    return {
        "like_behavior": {
            "favourites_count": favourites,
        },
        "posting_behavior": {
            "statuses_count": statuses,
        },
        "follow_behavior": {
            "followers_count": followers,
            "friends_count": friends,
            "listed_count": listed,
        },
        "profile_behavior": {
            "created_at": _raw_str(user, "created_at"),
            "verified": bool(user.get("verified", False)),
            "protected": bool(user.get("protected", False)),
            "default_profile_image": bool(user.get("default_profile_image", False)),
            "default_profile": bool(user.get("default_profile", False)),
            "url": _raw_str(user, "url"),
            "description": _raw_str(user, "description"),
            "location": _raw_str(user, "location"),
        },
        "comment_behavior": {
            "created_at": _raw_str(user, "created_at"),
            "statuses_count": statuses,
            "predicted_label": predicted_label,
        },
    }


def build_prompt_context(user: dict, prediction: dict) -> dict:
    tweets = user.get("tweets", [])
    if not isinstance(tweets, list):
        tweets = []

    tweet_samples = [
        item.strip()[:280]
        for item in tweets
        if isinstance(item, str) and item.strip()
    ][:5]

    return {
        "prediction": {
            "label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "probabilities": prediction.get("probabilities", {}),
        },
        "behavior_metrics": build_behavior_metrics(user, prediction=prediction),
        "profile": {
            "screen_name": _raw_str(user, "screen_name"),
            "name": _raw_str(user, "name"),
            "description": _raw_str(user, "description"),
            "location": _raw_str(user, "location"),
            "url": _raw_str(user, "url"),
        },
        "tweet_samples": tweet_samples,
    }
