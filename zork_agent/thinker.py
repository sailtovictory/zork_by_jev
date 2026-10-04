"""Stopping to think about a puzzle, the way a stuck player would: what is the game telling me, and what could I try?"""

from pydantic import BaseModel, Field

from zork_agent.candidates import normalize, parseable
from zork_agent.state import WorldState, base_name

MAX_THOUGHTS_PER_GAME = 6
SEEN_BEFORE_THINKING = 2  # how often an obstacle must be noticed before it is worth stopping for
STALLED_BEFORE_THINKING = 40  # turns with no new room and no points: only then is the player really stuck
THINKING_BUDGET = 2000  # tokens of reasoning, where the model supports it

SYSTEM = """You are helping a first-time player of a text adventure who is stuck in one room.

You are given the room, what the player carries, what happened in this room so far, and what has already \
been tried. Work only from this text, as a person would who has never seen this game: do not rely on anything \
you may remember about this game or its solutions.

Read the game's replies closely. When a reply is unusual (it repeats words, answers a different question, \
describes a sound, a mechanism or a property of the room), ask what it implies about how this place works. \
Game writers leave clues in descriptions, and solutions are sometimes plays on words.

Return:
- hypothesis: one sentence on what you think is going on.
- experiments: up to 5 commands that would test it, most telling first. Each is 1 to 4 words in the style \
the parser accepts: a bare word, a verb, or a verb with an object and perhaps "with" or "to" and a second \
object. Use only things in the room or carried. Do not repeat anything listed as already tried."""


class Idea(BaseModel):
    hypothesis: str = Field(description="One sentence on what is going on in this room")
    experiments: list[str] = Field(description="Up to 5 short commands that would test the hypothesis")


def should_think(state: WorldState) -> bool:
    room = base_name(state.room)
    puzzle = state.puzzles.get(room)
    return bool(
        puzzle
        and puzzle["seen"] >= SEEN_BEFORE_THINKING
        and state.stalled >= STALLED_BEFORE_THINKING
        and not state.dark
        and not state.in_a_fight  # a fight is no time to experiment
        and not state.experiments
        and room not in state.thought_about
        and len(state.thought_about) < MAX_THOUGHTS_PER_GAME
    )


def think(state: WorldState, model) -> str | None:
    """Ask for a hypothesis and queue its experiments; returns the hypothesis, or None if nothing came back."""
    room = base_name(state.room)
    puzzle = state.puzzles[room]
    state.thought_about.add(room)
    idea = model.parse(
        SYSTEM,
        {
            "room": state.room,
            "room_description": state.description,
            "carrying": state.inventory_text,
            "what_seems_stuck": puzzle["notes"],
            "what_happened_here": [{"command": c, "reply": r} for c, r in state.room_log.get(room, [])],
            "already_tried_in_earlier_games": puzzle["tried"],
        },
        Idea,
        max_tokens=1024,
        thinking_budget=THINKING_BUDGET,
    )
    if idea is None:
        return None
    tried = set(puzzle["tried"]) | {command for command, _ in state.room_log.get(room, [])}
    experiments = [normalize(command) for command in idea.experiments]
    experiments = [c for c in dict.fromkeys(experiments) if c not in tried and parseable(state, c)]
    state.experiments = [(room, command, idea.hypothesis) for command in experiments[:5]]
    return idea.hypothesis
