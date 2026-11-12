"""Ollama-backed language model (LangChain ``ChatOllama``)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from intent_agent.adapters.llm.host import normalize_host
from intent_agent.adapters.llm.langchain_model import LangChainLanguageModel


class ModelUnavailableError(RuntimeError):
    """Ollama is unreachable or the requested model has not been pulled."""


class OllamaLanguageModel(LangChainLanguageModel):
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
    ) -> None:
        self.model = model
        self.base_url = normalize_host(base_url)
        self.num_ctx = num_ctx
        super().__init__(self._make_chat_model)

    def _make_chat_model(self, max_tokens: int, json_mode: bool, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=self.num_ctx,
            **({"format": "json"} if json_mode else {}),
        )

    def _client(self):
        import ollama
        return ollama.Client(host=self.base_url)

    def list_models(self) -> list[str]:
        """Tags of the models pulled on the server ([] if unreachable)."""
        try:
            return [m.model for m in self._client().list().models]
        except Exception:
            return []

    def check_connection(self) -> None:
        """Raise ModelUnavailableError with a fix-it hint; otherwise warm the model."""
        try:
            self._client().show(self.model)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Cannot use Ollama model {self.model!r} at {self.base_url}: {exc}\n"
                f"Start the server with `ollama serve` and pull the model with "
                f"`ollama pull {self.model}`."
            ) from exc
        self.chat_model(1).invoke([HumanMessage(content="ok")])
