import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from utils.account_summary import build_account_metrics, build_behavior_metrics
from utils.llm_analysis import LLMAnalyzer
from utils.llm_config import LLMConfig


class AccountSummaryTests(unittest.TestCase):
    def test_build_account_metrics_includes_account_dimensions(self):
        user = {
            "followers_count": 100,
            "friends_count": 9,
            "listed_count": 3,
            "favourites_count": 42,
            "statuses_count": 250,
            "created_at": "2024-01-01T00:00:00Z",
            "verified": True,
            "protected": False,
            "default_profile_image": True,
            "default_profile": False,
            "tweets": ["hello", "", "world"],
        }

        metrics = build_account_metrics(
            user, now=datetime(2024, 1, 11, tzinfo=timezone.utc)
        )

        self.assertEqual(metrics["followers_count"], 100)
        self.assertEqual(metrics["friends_count"], 9)
        self.assertEqual(metrics["favourites_count"], 42)
        self.assertEqual(metrics["statuses_count"], 250)
        self.assertEqual(metrics["followers_friends_ratio"], 10.0)
        self.assertEqual(metrics["tweet_count"], 2)
        self.assertEqual(metrics["account_age_days"], 10)
        self.assertTrue(metrics["verified"])
        self.assertTrue(metrics["default_profile_image"])

    def test_build_behavior_metrics_groups_activity_dimensions(self):
        user = {
            "followers_count": 12,
            "friends_count": 4980,
            "listed_count": 0,
            "favourites_count": 60,
            "statuses_count": 120,
            "created_at": "2024-01-01T00:00:00Z",
            "default_profile_image": True,
            "default_profile": True,
            "tweets": ["promo", "follow me"],
        }

        metrics = build_behavior_metrics(
            user, now=datetime(2024, 1, 11, tzinfo=timezone.utc)
        )

        self.assertEqual(metrics["like_behavior"]["favourites_count"], 60)
        self.assertEqual(metrics["posting_behavior"]["statuses_count"], 120)
        self.assertEqual(metrics["follow_behavior"]["friends_count"], 4980)
        self.assertEqual(
            metrics["profile_behavior"]["created_at"], "2024-01-01T00:00:00Z"
        )
        self.assertEqual(
            metrics["comment_behavior"]["created_at"], "2024-01-01T00:00:00Z"
        )
        self.assertEqual(metrics["comment_behavior"]["statuses_count"], 120)

    def test_build_behavior_metrics_can_include_comment_prediction_context(self):
        user = {
            "statuses_count": 120,
            "created_at": "2024-01-01T00:00:00Z",
        }

        metrics = build_behavior_metrics(user, prediction={"label": "bot"})

        self.assertEqual(metrics["comment_behavior"]["created_at"], "2024-01-01T00:00:00Z")
        self.assertEqual(metrics["comment_behavior"]["statuses_count"], 120)
        self.assertEqual(metrics["comment_behavior"]["predicted_label"], "bot")


class LLMAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.user = {
            "followers_count": 10,
            "friends_count": 20,
            "favourites_count": 3,
            "statuses_count": 5,
            "tweets": ["sample tweet"],
        }
        self.prediction = {
            "label": "bot",
            "confidence": 0.82,
            "probabilities": {"human": 0.18, "bot": 0.82},
        }
        self.analyzer = LLMAnalyzer(
            LLMConfig(
                enabled=True,
                api_base="https://example.test/v1",
                model="test-model",
                api_key="test-key",
                max_retries=1,
            )
        )

    def test_parse_wrapped_json_response(self):
        parsed = self.analyzer._parse_response(
            '分析如下：{"risk_level":"high","summary":"风险较高",'
            '"behavior_anomalies":{"like_behavior":["点赞少"],'
            '"posting_behavior":["发帖数少"],'
            '"follow_behavior":["关注多"],'
            '"profile_behavior":["资料少"],'
            '"comment_behavior":["依据注册时间、发帖数和预测标签，发现互动异常问题"]},'
            '"key_factors":["发帖数少"],"content_signals":["样本少"],'
            '"recommendations":["人工复核"]}'
        )
        normalized = self.analyzer._normalize_result(
            parsed, self.user, self.prediction
        )

        self.assertEqual(normalized["status"], "success")
        self.assertEqual(normalized["risk_level"], "high")
        self.assertIn("模型预测结果为 bot", normalized["prediction_explanation"])
        self.assertEqual(normalized["summary"], "风险较高")
        self.assertIn("comment_behavior", normalized["behavior_anomalies"])
        self.assertEqual(
            normalized["behavior_anomalies"]["comment_behavior"],
            ["依据注册时间、发帖数和预测标签，发现互动异常问题"],
        )
        self.assertEqual(normalized["key_factors"], ["发帖数少"])
        self.assertIn("account_metrics", normalized)
        self.assertIn("behavior_metrics", normalized)

    def test_output_parameter_names_are_replaced_with_chinese_labels(self):
        parsed = self.analyzer._parse_response(
            '{"risk_level":"medium",'
            '"summary":"favourites_count 异常且 statuses_count 偏高",'
            '"behavior_anomalies":{"like_behavior":["favourites_count 很低"],'
            '"posting_behavior":["statuses_count 很高"],'
            '"follow_behavior":["friends_count 明显高于 followers_count"],'
            '"profile_behavior":["created_at 较新"],'
            '"comment_behavior":["comment_behavior 只能依据 created_at 和 statuses_count 分析，predicted_label 为 bot"]},'
            '"key_factors":["listed_count 很低"],'
            '"content_signals":["样本含导流"],'
            '"recommendations":["复核 screen_name"]}'
        )

        normalized = self.analyzer._normalize_result(
            parsed, self.user, self.prediction
        )

        rendered = " ".join(
            [
                normalized["summary"],
                *normalized["behavior_anomalies"]["like_behavior"],
                *normalized["behavior_anomalies"]["posting_behavior"],
                *normalized["behavior_anomalies"]["follow_behavior"],
                *normalized["behavior_anomalies"]["profile_behavior"],
                *normalized["behavior_anomalies"]["comment_behavior"],
                *normalized["key_factors"],
                *normalized["content_signals"],
                *normalized["recommendations"],
            ]
        )
        self.assertNotIn("favourites_count", rendered)
        self.assertNotIn("statuses_count", rendered)
        self.assertNotIn("friends_count", rendered)
        self.assertNotIn("comment_behavior", rendered)
        self.assertNotIn("predicted_label", rendered)
        self.assertNotIn("间接判断", rendered)
        self.assertNotIn("间接解析", rendered)
        self.assertIn("点赞数", normalized["summary"])
        self.assertIn("发帖数", normalized["summary"])
        self.assertIn("评论行为", rendered)
        self.assertIn("预测标签", rendered)

    def test_indirect_comment_wording_is_removed_from_output(self):
        parsed = self.analyzer._parse_response(
            '{"risk_level":"medium","summary":"模型预测结果为 bot，需复核",'
            '"behavior_anomalies":{"comment_behavior":["依据注册时间和发帖数间接判断存在问题"]},'
            '"key_factors":[],"content_signals":[],"recommendations":[]}'
        )

        normalized = self.analyzer._normalize_result(
            parsed, self.user, self.prediction
        )
        rendered = " ".join(normalized["behavior_anomalies"]["comment_behavior"])

        self.assertNotIn("间接判断", rendered)
        self.assertIn("依据注册时间和发帖数分析存在问题", rendered)

    def test_risk_level_is_forced_to_prediction_probability(self):
        parsed = self.analyzer._parse_response(
            '{"risk_level":"low","summary":"模型预测结果为 bot，风险较低",'
            '"behavior_anomalies":{"comment_behavior":["间接判断"]},'
            '"key_factors":[],"content_signals":[],"recommendations":[]}'
        )

        normalized = self.analyzer._normalize_result(
            parsed, self.user, self.prediction
        )

        self.assertEqual(normalized["risk_level"], "high")

    def test_build_prompt_requires_behavior_anomaly_analysis(self):
        prompt = self.analyzer._build_prompt(self.user, self.prediction)

        self.assertIn("账号异常行为", prompt)
        self.assertIn("like_behavior", prompt)
        self.assertIn("comment_behavior", prompt)
        self.assertIn("评论行为", prompt)
        self.assertIn("注册时间", prompt)
        self.assertIn("发帖数", prompt)
        self.assertIn("实际预测标签", prompt)
        self.assertIn("没有评论数、回复数或评论内容字段", prompt)
        self.assertIn("发现", prompt)
        self.assertIn("未发现明显问题", prompt)
        self.assertIn("不要输出判断方式本身的元说明", prompt)
        self.assertNotIn("间接判断", prompt)
        self.assertNotIn("间接解析", prompt)
        self.assertNotIn("评论/回复", prompt)
        self.assertNotIn("data_available", prompt)
        self.assertNotIn("source_field", prompt)
        self.assertNotIn("likes_per_day", prompt)
        self.assertNotIn("likes_per_post", prompt)
        self.assertNotIn("comments_per_day", prompt)
        self.assertNotIn("posts_per_day", prompt)
        self.assertNotIn("followers_friends_ratio", prompt)
        self.assertNotIn("account_age_days", prompt)

    @patch("requests.post")
    def test_call_chat_completions_does_not_set_token_or_timeout_limits(
        self, mock_post: Mock
    ):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"summary":"ok"}'}}]
        }
        mock_post.return_value = response

        content = self.analyzer._call_chat_completions("prompt")

        self.assertEqual(content, '{"summary":"ok"}')
        _, kwargs = mock_post.call_args
        self.assertNotIn("timeout", kwargs)
        self.assertNotIn("max_tokens", kwargs["json"])

    def test_invalid_json_response_raises(self):
        with self.assertRaises(ValueError):
            self.analyzer._parse_response("not json")

    def test_disabled_llm_returns_skipped_analysis(self):
        analyzer = LLMAnalyzer(
            LLMConfig(
                enabled=False,
                api_base="https://example.test/v1",
                model="test-model",
                api_key="",
                max_retries=1,
            )
        )

        analysis = analyzer.analyze_prediction(self.user, self.prediction)

        self.assertEqual(analysis["status"], "skipped")
        self.assertEqual(analysis["risk_level"], "high")
        self.assertIn("模型预测结果为 bot", analysis["prediction_explanation"])
        self.assertIn("comment_behavior", analysis["behavior_anomalies"])
        self.assertIn("account_metrics", analysis)


if __name__ == "__main__":
    unittest.main()
