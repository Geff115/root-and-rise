"""
shared/llm_client.py
--------------------
Thin wrapper around the Groq API.

Supported models (free tier as of 2026):
  - llama-3.3-70b-versatile   ← default, best quality
  - llama-3.1-8b-instant      ← faster, good for iteration
  - mixtral-8x7b-32768        ← strong on structured tasks
  - gemma2-9b-it              ← lightweight alternative

Usage:
    from shared.llm_client import LLMClient
    llm = LLMClient()
    response = llm.chat("Write a review for a spicy restaurant")
"""

import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Default system prompt — establishes the agent's identity
DEFAULT_SYSTEM_PROMPT = """You are an intelligent user behaviour modeling and recommendation agent.
You reason carefully before producing outputs and always ground your responses in the
user's behavioural history and context provided to you.
When contextually appropriate, you naturally incorporate Nigerian English expressions
and local cultural references."""


class LLMClient:
    """
    Wrapper around Groq's chat completion API.
    Handles model selection, temperature control, and structured output extraction.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def chat(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Single-turn chat completion.
        Returns the assistant's response as a plain string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_override or self.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )
        return response.choices[0].message.content.strip()

    def chat_with_history(
        self,
        messages: list[dict],
        system_override: Optional[str] = None,
    ) -> str:
        """
        Multi-turn chat completion.
        messages: list of {"role": "user"/"assistant", "content": "..."}
        Returns the assistant's response as a plain string.
        """
        full_messages = [
            {"role": "system", "content": system_override or self.system_prompt}
        ] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content.strip()

    def structured_chat(self, user_message: str, system_override: Optional[str] = None) -> dict:
        """
        Chat completion expecting a JSON response.
        Adds JSON enforcement to the prompt and parses the result.
        """
        import json
        import re

        json_instruction = (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No preamble, no explanation, no markdown code fences."
        )

        raw = self.chat(
            user_message + json_instruction,
            system_override=system_override,
            temperature=0.3,  # Lower temp for structured outputs
        )

        # Strip markdown fences if the model adds them anyway
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON.\nRaw output:\n{raw}\nError: {e}"
            )

    def switch_model(self, model: str):
        """Hot-swap the model mid-session."""
        self.model = model
        print(f"[LLMClient] Switched to model: {model}")