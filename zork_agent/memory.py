"""Long-term memory across games.

The map is kept exactly as the harness observed it. Lessons are written by the text-reading model, which rereads the
transcript after each attempt.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from zork_agent.state import WorldState

MAX_LESSONS = 25
RESPONSE_LENGTH = 300  # characters of each game response kept in the transcript sent for reflection

SYSTEM = f"""You maintain the long-term memory of an autonomous player of the text adventure Zork I.

You are given the lessons learned so far and the transcript of the latest attempt. Return the full, updated \
list of lessons (at most {MAX_LESSONS}), most valuable first. The player reads this list before every \
decision in its next attempt, so each lesson must stand on its own and name exact rooms and exact commands.

Worth keeping:
- command sequences that raised the score, with the room they happen in and the points gained
- what killed the player or lost points, and what to do instead
- which items turned out to be needed where (a light source before a dark room, a weapon before a fight)
- commands and areas that wasted many turns with nothing gained
- limits the game revealed, such as a lamp running low: note the turn it happened and what it means for next time

Every game starts from the beginning: the player carries nothing, the trophy case is empty, and every object is back where it started. Never write that something is already held, already deposited or already done; write what has to be done again each game.

Do not record routes between rooms: the player keeps an exact map separately. A room name followed by a \
number, such as "Forest (2)", is a different room from the one without it.

If the player followed a lesson and still died or lost points, that lesson is wrong: correct it or delete it. State a cause or a cure only when the game's own text in the transcript says so; do not guess at what would have helped.

Never drop a lesson about a sequence that raised the score unless the new transcript contradicts it; \
when the list is full, drop the least valuable lesson about wasted turns first. Merge lessons that say \
the same thing. Record only what the transcripts show; do not add knowledge of Zork from anywhere else."""


class Lessons(BaseModel):
    lessons: list[str] = Field(description="The full updated list of lessons, most valuable first")


def load(path: Path) -> dict:
    empty = {"attempts": 0, "lessons": [], "map": {}, "blocked": [], "signatures": {}, "dark_rooms": [], "deadly_rooms": [], "treasures": [], "trophy": {}, "ambiguous": [], "puzzles": {}, "lit_rooms": [], "enemies": {}}
    return empty | json.loads(path.read_text(encoding="utf-8")) if path.exists() else empty


def _save(path: Path, memory: dict) -> None:
    path.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def recall(path: Path, state: WorldState) -> None:
    """Start a game knowing the map, the dead ends and the lessons from earlier games."""
    memory = load(path)
    state.lessons = memory["lessons"]
    state.map = memory["map"]
    state.signatures = memory["signatures"]
    state.blocked = {(room, direction) for room, direction in memory["blocked"]}
    state.dark_rooms = set(memory["dark_rooms"])
    state.deadly_rooms = set(memory["deadly_rooms"])
    state.treasures = set(memory["treasures"])
    state.trophy = memory["trophy"]
    state.ambiguous = set(memory["ambiguous"])
    state.puzzles = memory["puzzles"]
    state.enemies = memory["enemies"]
    state.lit_rooms |= set(memory["lit_rooms"])


def save_world(path: Path, state: WorldState) -> None:
    """Store the map as observed. A direction that led somewhere at any point is not a dead end."""
    memory = load(path)
    # Rooms told apart by dropped items are left out: the items land in different rooms every game.
    memory["map"] = {
        room: {action: destination for action, destination in exits.items() if "[" not in destination}
        for room, exits in state.map.items() if "[" not in room
    }
    memory["treasures"] = sorted(state.treasures)
    memory["trophy"] = state.trophy
    memory["ambiguous"] = sorted(state.ambiguous)
    memory["puzzles"] = state.puzzles
    memory["enemies"] = state.enemies
    memory["lit_rooms"] = sorted(room for room in state.lit_rooms if "[" not in room)
    memory["signatures"] = state.signatures
    memory["dark_rooms"] = sorted(state.dark_rooms)
    memory["deadly_rooms"] = sorted(state.deadly_rooms)
    memory["blocked"] = sorted(pair for pair in state.blocked if pair[1] not in state.map.get(pair[0], {}))
    _save(path, memory)


def _transcript(records: list[dict], response_length: int) -> list[dict]:
    return [
        {
            "step": r["step"],
            "command": r.get("action"),
            "response": r["response"][:response_length],
            "room_after": r["room"],
            "score_after": r["score"],
        }
        for r in records
    ]


def reflect(path: Path, records: list[dict], reader) -> list[str]:
    """Fold one attempt's transcript into the lessons and return the new lessons."""
    memory = load(path)
    response_length = RESPONSE_LENGTH
    payload = {"lessons_so_far": memory["lessons"], "transcript": _transcript(records, response_length)}
    # A local model has a small context: shorten the quoted responses until the transcript fits.
    while reader.max_input_chars and len(json.dumps(payload)) > reader.max_input_chars and response_length > 40:
        response_length = response_length * 2 // 3
        payload["transcript"] = _transcript(records, response_length)
    result = reader.parse(SYSTEM, payload, Lessons, max_tokens=4096)
    if result is None:  # refusal or truncated output: keep what we had
        return memory["lessons"]
    memory["lessons"] = result.lessons[:MAX_LESSONS]
    memory["attempts"] += 1
    _save(path, memory)
    return memory["lessons"]
