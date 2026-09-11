import re
from typing import Optional

from core.types import RouteDecision


class Router:
    """
        Router simples e determinístico.

        Objetivo:
        - evitar mandar tudo para a LLM
        - decidir quando usar resposta direta
        - decidir quando chamar uma tool específica
        - mandar para LLM apenas quando necessário
        """

    def __init__(self) -> None:
        """
        A ordem importa, por tanto eu vou usar primeiro os KBs
        """
        self.tool_patterns = {
            "kblocal": [
                r"\bslo\b",
                r"\bsre\b",
                r"\bo que faz um sre\b",
                r"\bOpen Finance\b",
                r"\bsla\b",
            ],
            "wikipedia": [
                r"\bquem foi\b",
                r"\bquem é\b",
                r"\bo que é\b",
                r"\bdefina\b",
            ],
        }

        self.direct_patterns = [
            r"\boi\b",
            r"\bol[áa]\b",
            r"\bvaleu\b",
            r"\bobrigad[ao]\b",
            r"\btchau\b",
        ]

    def route(self, user_input: str) -> RouteDecision:
        """
        Defines the route, whether it is "tool", "llm" or "direct"

        Avoids increase of latency and unnecessary LLM invocations. Hence, we have a best price solution :D
        :param user_input:
        :return RouterDecision:
        """
        cleaned_input = self._normalize(user_input)

        if not cleaned_input:
            return RouteDecision(
                route="direct_response",
                reason="entrada_vazia",
                direct_response="Pode me mandar sua pergunta."
            )

        direct_response = self._try_direct_response(cleaned_input)
        if direct_response:
            return direct_response

        tool_decision = self._try_tool_route(cleaned_input)
        if tool_decision:
            return tool_decision

        return RouteDecision(
            route="agent",
            reason="fallback_to_agent",
            metadata={
                "original_input": user_input,
                "normalized_input": cleaned_input,
                "confidence": "medium",
            }
        )

    def _try_direct_response(self, text: str) -> Optional[RouteDecision]:
        if any(re.search(pattern, text) for pattern in self.direct_patterns):
            if re.fullmatch(r"\b(oi|olá|ola)\b", text):
                return RouteDecision(
                    route="direct_response",
                    reason="saudacao_simples",
                    direct_response="Oi! como posso te ajudar?",
                )

            if re.search(r"\b(valeu|obrigad[ao])\b", text):
                return RouteDecision(
                    route="direct_response",
                    reason="agradecimento_simples",
                    direct_response="De nada! Pode me mandar a próxima",
                )

            if re.search(r"\btchau\b", text):
                return RouteDecision(
                    route="direct_response",
                    reason="encerramento_simples",
                    direct_response="Até mais!",
                )
        return None

    def _try_tool_route(self, text: str) -> Optional[RouteDecision]:
        for tool_name, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return self._build_tool_decision(tool_name, text, pattern)
        return None

    def _build_tool_decision(self, tool_name: str, text: str, matched_pattern: str) -> RouteDecision:
        if tool_name == "kblocal":
            return RouteDecision(
                route="tool",
                reason="consulta_informacional_local_detectada",
                tool_name="search_local_kb",
                tool_input={"query": text},
                metadata={
                    "matched_pattern": matched_pattern,
                    "confidence": "high",
                },
            )

        if tool_name == "wikipedia":
            query = self._extract_topic_for_wikipedia(text)
            return RouteDecision(
                route="tool",
                reason="consulta_informacional_detectada",
                tool_name="wikipedia_search",
                tool_input={"query": query},
                metadata={
                    "matched_pattern": matched_pattern,
                    "confidence": "medium",
                }
            )

        return RouteDecision(
            route="llm",
            reason="tool_detectada_mas_sem_builder_especifico",
            metadata={
                "tool_name": tool_name,
                "matched_pattern": matched_pattern,
            },
        )

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_topic_for_wikipedia(self, text: str) -> str:
        prefixes = [
            "o que é", "quem foi", "quem é",
            "defina", "wiki", "wikipedia"
        ]

        cleaned = text.strip()
        lowered = cleaned.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                return cleaned[len(prefix):].strip(" :,-")

        return cleaned

    def log_decision(self, user_input: str, decision: RouteDecision) -> None:
        print(
            {
                "input": user_input,
                "route": decision.route,
                "reason": decision.reason,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
                "metadata": decision.metadata,
            }
        )
