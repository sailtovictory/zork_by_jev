"""The text-reading model: one call in, one schema-shaped object out. Haiku over the API, or a local model in Ollama."""

import json
import os
import urllib.request

from pydantic import BaseModel, ValidationError

HAIKU_MODEL = "claude-haiku-4-5"
LOCAL_MODEL = "gemma4:12b"
LOCAL_HOST = "http://localhost:11434"
LOCAL_CONTEXT = 16384  # tokens; what fits beside a 14B model in 16 GB of VRAM


class HaikuReader:
    """Reads ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID, if the key needs one) from the environment."""

    name = HAIKU_MODEL
    max_input_chars = None

    def __init__(self):
        import anthropic

        # Keys that are not scoped to a workspace must name one on every request.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self.client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)

    def parse(self, system: str, payload: dict, schema: type[BaseModel], max_tokens: int) -> BaseModel | None:
        response = self.client.messages.parse(
            model=HAIKU_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload)}],
            output_format=schema,
        )
        return response.parsed_output  # None on a refusal or truncated output


class LocalReader:
    """A model served by Ollama on this machine. OLLAMA_MODEL and OLLAMA_HOST override the defaults."""

    # Ollama silently drops the start of a prompt that overflows the context, so callers trim to this.
    max_input_chars = LOCAL_CONTEXT * 3

    def __init__(self, model: str | None = None):
        self.name = model or os.environ.get("OLLAMA_MODEL", LOCAL_MODEL)
        host = os.environ.get("OLLAMA_HOST", LOCAL_HOST)
        self.url = (host if host.startswith("http") else f"http://{host}").rstrip("/") + "/api/chat"

    def parse(self, system: str, payload: dict, schema: type[BaseModel], max_tokens: int) -> BaseModel | None:
        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,  # reasoning models would otherwise spend seconds thinking before every turn
            "options": {"temperature": 0, "num_ctx": LOCAL_CONTEXT, "num_predict": max_tokens},
        }
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            content = json.load(response)["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError:  # output cut off at num_predict
            return None
