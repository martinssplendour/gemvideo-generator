from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from schemas import utc_now_iso


class SafetyCheckService:
    PROFILES = {
        "standard": {
            "impersonation": "warning",
            "trademark": "warning",
            "unsafe_content": "warning",
            "fraud": "warning",
        },
        "strict_brand": {
            "impersonation": "high",
            "trademark": "high",
            "unsafe_content": "high",
            "fraud": "warning",
        },
        "kids": {
            "impersonation": "warning",
            "trademark": "warning",
            "unsafe_content": "high",
            "fraud": "high",
        },
    }

    IMPERSONATION_TERMS = [
        "as if i am",
        "exactly like",
        "clone voice",
        "impersonate",
        "deepfake",
        "sound exactly like",
    ]
    TRADEMARK_TERMS = [
        "nike",
        "coca-cola",
        "tesla",
        "apple",
        "google",
        "disney",
    ]
    UNSAFE_TERMS = [
        "self-harm",
        "violence",
        "illegal weapon",
        "hate speech",
        "terrorist",
        "bomb tutorial",
    ]
    FRAUD_TERMS = [
        "guaranteed returns",
        "fake testimonial",
        "forged",
        "steal account",
    ]

    def __init__(
        self,
        profile: str = "standard",
        mode: str = "warn_only",
        model_classifier: Optional[Any] = None,
    ):
        self.profile = profile if profile in self.PROFILES else "standard"
        self.mode = mode
        self.model_classifier = model_classifier

    def run_checks(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        text_nodes = self._collect_text_nodes(manifest)
        rules = self.PROFILES.get(self.profile, self.PROFILES["standard"])

        for path, text in text_nodes:
            lower = text.lower()
            warnings.extend(
                self._match_terms(lower, path, self.IMPERSONATION_TERMS, "impersonation", rules)
            )
            warnings.extend(
                self._match_terms(lower, path, self.TRADEMARK_TERMS, "trademark", rules)
            )
            warnings.extend(
                self._match_terms(lower, path, self.UNSAFE_TERMS, "unsafe_content", rules)
            )
            warnings.extend(self._match_terms(lower, path, self.FRAUD_TERMS, "fraud", rules))

        model_warnings = self._run_model_classifier(manifest, rules)
        warnings.extend(model_warnings)

        dedup = []
        seen = set()
        for warning in warnings:
            key = (warning["category"], warning["field_path"], warning["message"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(warning)
        return dedup

    def _run_model_classifier(
        self, manifest: Dict[str, Any], rules: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        if not self.model_classifier:
            return []
        try:
            result = self.model_classifier.classify_manifest(manifest)
            warnings = []
            for item in result:
                category = item.get("category")
                if category not in rules:
                    continue
                warnings.append(
                    {
                        "category": category,
                        "severity": rules.get(category, "warning"),
                        "message": item.get("message", "Model flagged potential policy issue."),
                        "field_path": item.get("field_path", "manifest"),
                        "source": "model_classifier",
                        "timestamp": utc_now_iso(),
                    }
                )
            return warnings
        except Exception:
            return []

    def _match_terms(
        self,
        lower_text: str,
        field_path: str,
        terms: List[str],
        category: str,
        rules: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        matches = []
        for term in terms:
            if term in lower_text:
                matches.append(
                    {
                        "category": category,
                        "severity": rules.get(category, "warning"),
                        "message": f"Potential {category} risk: found '{term}'.",
                        "field_path": field_path,
                        "source": "rule_based",
                        "policy_profile": self.profile,
                        "timestamp": utc_now_iso(),
                    }
                )
        return matches

    def _collect_text_nodes(self, data: Any, path: str = "") -> List[tuple[str, str]]:
        nodes: List[tuple[str, str]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                next_path = f"{path}.{key}" if path else key
                nodes.extend(self._collect_text_nodes(value, next_path))
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                next_path = f"{path}[{idx}]"
                nodes.extend(self._collect_text_nodes(value, next_path))
        elif isinstance(data, str):
            nodes.append((path, data))
        return nodes


class GeminiSafetyClassifier:
    def __init__(self, client: Any, model_name: str):
        self.client = client
        self.model_name = model_name

    def classify_manifest(self, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
        payload = {
            "brief": manifest.get("brief", {}),
            "world": manifest.get("world", {}),
            "scenes": manifest.get("scenes", []),
            "characters": manifest.get("characters", []),
        }
        prompt = (
            "Analyze this video manifest for risks in categories: impersonation, trademark, "
            "unsafe_content, fraud. Return JSON list with fields category, message, field_path. "
            f"Manifest: {json.dumps(payload, ensure_ascii=False)}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = getattr(response, "text", "") or "[]"
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except Exception:
            return []
