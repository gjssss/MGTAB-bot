import math
import numpy as np

PROPERTY_DIM = 20

# Column layout of the feature vector:
#   col[0]:  is_protected
#   col[1]:  is_verified
#   col[2]:  default_profile_image
#   col[3]:  followers_count (ln+minmax normalized)
#   col[4]:  geo_enabled
#   col[5]:  friends_count (ln+minmax normalized)
#   col[6]:  listed_count (ln+minmax normalized)
#   col[7]:  created_at (ln+minmax normalized)
#   col[8]:  contributors_enabled
#   col[9]:  favourites_count (ln+minmax normalized)
#   col[10]: statuses_count (ln+minmax normalized)
#   col[11]: screen_name_length (minmax normalized)
#   col[12]: name_length (minmax normalized)
#   col[13]: description_length (minmax normalized)
#   col[14]: followers_friends_ratios (ln+minmax normalized)
#   col[15]: default_profile
#   col[16]: has_url
#   col[17]: has_description
#   col[18]: has_location
#   col[19]: unknown_bool (default 0)
#   col[20..]: Qwen3-Embedding (dimension depends on selected size:
#              0.6B → 1024, 4B → 2560, 8B → 4096)

NORMALIZATION_CONFIG = {
    "followers_count":          {"col": 3,  "min": 0.0,      "max": 25.572674, "log": True},
    "friends_count":            {"col": 5,  "min": 0.0,      "max": 21.029877, "log": True},
    "listed_count":             {"col": 6,  "min": 0.0,      "max": 17.675406, "log": True},
    "created_at":               {"col": 7,  "min": 36.553529,"max": 51.711108, "log": True},
    "favourites_count":         {"col": 9,  "min": 0.0,      "max": 19.711042, "log": True},
    "statuses_count":           {"col": 10, "min": 0.0,      "max": 20.386231, "log": True},
    "screen_name_length":       {"col": 11, "min": 3.0,      "max": 15.0,      "log": False},
    "name_length":              {"col": 12, "min": 1.0,      "max": 50.0,      "log": False},
    "description_length":       {"col": 13, "min": 0.0,      "max": 204.0,     "log": False},
    "followers_friends_ratios": {"col": 14, "min": 0.0,      "max": 11.169299, "log": True},
}

BOOL_FIELDS = {
    "protected":              0,
    "verified":               1,
    "default_profile_image":  2,
    "geo_enabled":            4,
    "contributors_enabled":   8,
    "default_profile":        15,
    "has_url":                16,
    "has_description":        17,
    "has_location":           18,
}


def _minmax_normalize(value, vmin, vmax):
    if vmax == vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def extract_properties(user_json: dict) -> np.ndarray:
    """Extract 20-dim property vector from user JSON."""
    vec = np.zeros(PROPERTY_DIM, dtype=np.float32)

    vec[BOOL_FIELDS["protected"]] = float(bool(user_json.get("protected", False)))
    vec[BOOL_FIELDS["verified"]] = float(bool(user_json.get("verified", False)))
    vec[BOOL_FIELDS["default_profile_image"]] = float(bool(user_json.get("default_profile_image", False)))
    vec[BOOL_FIELDS["geo_enabled"]] = float(bool(user_json.get("geo_enabled", False)))
    vec[BOOL_FIELDS["contributors_enabled"]] = float(bool(user_json.get("contributors_enabled", False)))
    vec[BOOL_FIELDS["default_profile"]] = float(bool(user_json.get("default_profile", False)))
    vec[BOOL_FIELDS["has_url"]] = float(bool(user_json.get("url")))
    vec[BOOL_FIELDS["has_description"]] = float(bool(user_json.get("description")))
    vec[BOOL_FIELDS["has_location"]] = float(bool(user_json.get("location")))
    vec[19] = 0.0

    followers = user_json.get("followers_count", 0)
    friends = user_json.get("friends_count", 0)

    for field_name, cfg in NORMALIZATION_CONFIG.items():
        col_idx = cfg["col"]
        if field_name == "followers_friends_ratios":
            raw = followers / (friends + 1)
        elif field_name == "screen_name_length":
            raw = len(user_json.get("screen_name", ""))
        elif field_name == "name_length":
            raw = len(user_json.get("name", ""))
        elif field_name == "description_length":
            raw = len(user_json.get("description", "") or "")
        elif field_name == "created_at":
            raw = _parse_created_at(user_json.get("created_at"))
        else:
            raw = user_json.get(field_name, 0)

        if cfg["log"]:
            raw = math.log(raw + 1)

        vec[col_idx] = _minmax_normalize(raw, cfg["min"], cfg["max"])

    return vec


def _parse_created_at(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    from datetime import datetime, timezone

    dt = None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%a %b %d %H:%M:%S %z %Y"):
        try:
            dt = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        return 0.0

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


def build_feature_vector(user_json: dict, tweet_embedding: np.ndarray) -> np.ndarray:
    """Build the full (PROPERTY_DIM + embedding_dim)-dim feature vector.

    The embedding dimension is determined by the Qwen3-Embedding size selected
    at training time and persisted in checkpoints/preprocess_config.json.
    """
    if tweet_embedding is None:
        raise ValueError("tweet_embedding must not be None")
    if tweet_embedding.ndim != 1:
        raise ValueError(
            f"Expected 1-D tweet_embedding, got shape {tweet_embedding.shape}"
        )
    props = extract_properties(user_json)
    return np.concatenate([props, tweet_embedding]).astype(np.float32)
