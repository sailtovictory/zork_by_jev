"""Policies: given the state and candidate actions, pick one."""

import random

from zork_agent.state import WorldState

INSTRUCTIONS = (
    "Which command should the player type next to make the most progress in Zork I? "
    "Prefer exploring new rooms, picking up useful items and treasures, and solving puzzles. "
    "Avoid repeating commands that already failed and avoid pacing between visited rooms."
)


class RandomPolicy:
    """Baseline that needs no API key."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, state: WorldState, candidates: dict[str, str | None]) -> tuple[str, dict]:
        return self.rng.choice(list(candidates)), {}


class JevPolicy:
    """One Choice question per turn; the action is sampled from Jev's probabilities.

    The client reads TYPESAFE_API_KEY, and TYPESAFE_BASE_URL if set, so the same
    code runs against hosted Jev or a wire-compatible OpenJev server.
    """

    def __init__(self, rng: random.Random, model: str | None = None, greedy: bool = False, temperature: float = 1.0):
        from typesafe_sdk import TypeSafeClient

        self.client = TypeSafeClient(model=model)
        self.rng = rng
        self.greedy = greedy
        self.temperature = temperature

    def choose(self, state: WorldState, candidates: dict[str, str | None]) -> tuple[str, dict]:
        from typesafe_sdk import Choice

        response = self.client.system_one(
            state=state.summary(),
            questions={"action": Choice(instructions=INSTRUCTIONS, criteria=candidates)},
        )
        answer = response.choices["action"]
        action = answer.choice
        if not self.greedy:
            actions = list(answer.probabilities)
            weights = [answer.probabilities[a] ** (1 / self.temperature) for a in actions]
            if sum(weights) > 0:
                action = self.rng.choices(actions, weights)[0]
        top = sorted(answer.probabilities.items(), key=lambda item: -item[1])[:5]
        return action, {
            "confidence": answer.confidence,
            "top": {name: round(p, 4) for name, p in top},
            "input_tokens": response.usage.input_tokens,
        }
