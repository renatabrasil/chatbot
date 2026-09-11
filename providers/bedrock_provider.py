from __future__ import annotations

from typing import List, Optional, Tuple

import boto3

from core.types import Message, ModelMetrics


class BedrockProvider:
    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 800,
        profile_name: Optional[str] = None,
    ) -> None:
        self.model = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=region_name)

    def chat(self, messages: List[Message]) -> Tuple[str, ModelMetrics]:
        system_texts = []
        bedrock_messages = []

        for m in messages:
            if m.role == "system":
                system_texts.append({"text": m.content})
                continue

            role = "assistant" if m.role == "assistant" else "user"
            bedrock_messages.append(
                {
                    "role": role,
                    "content": [{"text": m.content}],
                }
            )

        response = self.client.converse(
            modelId=self.model,
            system=system_texts if system_texts else None,
            messages=bedrock_messages,
            inferenceConfig={
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
                "topP": 0.9,
            },
        )

        output_message = response["output"]["message"]
        answer_parts = []

        for item in output_message.get("content", []):
            if "text" in item:
                answer_parts.append(item["text"])

        answer = "\n".join(answer_parts).strip()

        usage = response.get("usage", {})
        metrics = response.get("metrics", {})

        model_metrics = ModelMetrics(
            latency_ms=float(metrics.get("latencyMs", 0.0)),
            prompt_tokens=usage.get("inputTokens"),
            completion_tokens=usage.get("outputTokens"),
            total_duration_ms=float(metrics.get("latencyMs", 0.0)),
        )

        return answer, model_metrics