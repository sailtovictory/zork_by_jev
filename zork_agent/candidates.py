"""Candidate actions. Jev only picks; whatever proposes the candidates decides what it can pick from."""

import re

from zork_agent.state import DIRECTIONS, WorldState
from zork_agent.vocab import WORD_LENGTH

MAX_CANDIDATES = 255  # Choice accepts at most 255 options

MAX_VISITS = 3  # a room entered this often is not offered as a destination while a fresher known exit remains
MAX_EXIT_USES = 3  # an exit taken this often from one room gives way to probing directions not yet tried
ROOM_VERBS = ["examine", "take", "open", "close", "read", "move", "push", "pull", "climb", "enter", "turn on", "eat", "drink"]
INVENTORY_VERBS = ["examine", "open", "read", "drop", "turn on", "turn off", "eat"]
META = {"look", "l", "g", "again", "save", "restore", "restart", "quit", "q", "script", "unscript", "score", "inventory", "i", "verbose", "brief", "superbrief", "version", "diagnose"}
TWO_OBJECT = ["put {held} in {seen}", "attack {seen} with {held}", "unlock {seen} with {held}", "tie {held} to {seen}"]

ABBREVIATIONS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "u": "up", "d": "down"}
# Phrasings the parser treats alike, rewritten to one form so they do not split the model's vote.
SYNONYMS = [
    (r"^(?:go|walk|run|head|climb) (?=(?:%s)$)" % "|".join(DIRECTIONS), ""),
    (r"^(?:get|grab|pick up) ", "take "),
    (r"^(?:x|inspect|look at) ", "examine "),
    (r"^light ", "turn on "),
]


def normalize(command: str) -> str:
    """Lower-case a command, drop articles and collapse synonyms to a single canonical phrasing."""
    words = [word for word in re.sub(r"[^a-z0-9 ]", " ", command.lower()).split() if word not in ("the", "a", "an")]
    command = " ".join(ABBREVIATIONS.get(word, word) if len(words) == 1 else word for word in words)
    for pattern, replacement in SYNONYMS:
        command = re.sub(pattern, replacement, command)
    return ABBREVIATIONS.get(command, command)


def movement(state: WorldState) -> dict[str, str | None]:
    """Compass directions plus every action already known to lead somewhere from this room."""
    candidates: dict[str, str | None] = {direction: "untried direction, not known to be an exit" for direction in DIRECTIONS}
    for action, destination in state.map.get(state.room, {}).items():
        candidates[action] = f"leads to {destination} (visited {state.visits.get(destination, 0)} times this game)"
    return candidates


def stale_exits(state: WorldState) -> set[str]:
    """Known exits not worth offering right now.

    A known exit always beats guessing at compass directions, so guesses never hide one. Exits into
    much-visited rooms give way to fresher known exits; an exit taken several times from this room
    gives way to untried directions; and when every way out is well trodden the agent rotates through
    them, which is also what gets it through the maze, where every room looks the same.
    """
    exits = state.map.get(state.room, {})
    visits = {action: state.visits.get(destination, 0) for action, destination in exits.items()}
    uses = {action: state.uses.get((state.room, action), 0) for action in exits}
    fresh = {action for action in exits if visits[action] < MAX_VISITS and uses[action] < MAX_EXIT_USES}
    if fresh:
        return set(exits) - fresh
    if any(d not in exits and not state.was_tried(d) for d in DIRECTIONS):
        return {action for action in exits if uses[action] >= MAX_EXIT_USES}
    rank = {action: (visits[action], uses[action]) for action in exits}
    return {action for action in rank if rank[action] > min(rank.values())}


def parseable(state: WorldState, command: str) -> bool:
    """True for a game command (not save, quit and the like) made only of words in the game's dictionary."""
    words = re.findall(r"[a-z]+", command.lower())
    if not words or words[0] not in META and {"self", "me", "myself"} & set(words):
        return False  # empty, or aimed at the player ("kill self with sword")
    return words[0] not in META and all(word[:WORD_LENGTH] in state.dictionary for word in words)


def finalize(state: WorldState, candidates: dict[str, str | None]) -> dict[str, str | None]:
    """Drop actions already tried here and stale exits; if nothing is left, fall back to the known ways out."""
    stale = stale_exits(state)
    kept = {action: hint for action, hint in candidates.items() if not state.was_tried(action) and action not in stale}
    exits = movement(state)
    kept = kept or {action: exits[action] for action in state.map.get(state.room, {})} or {"look": None}
    return dict(list(kept.items())[:MAX_CANDIDATES])


class TemplateProposer:
    """Verb templates crossed with the dictionary nouns found in the room and inventory text."""

    def propose(self, state: WorldState) -> dict[str, str | None]:
        held = state.inventory_nouns
        seen = [noun for noun in state.room_nouns if noun not in held]
        candidates = movement(state)
        candidates |= {f"{verb} {noun}": None for noun in seen for verb in ROOM_VERBS}
        candidates |= {f"{verb} {noun}": None for noun in held for verb in INVENTORY_VERBS}
        candidates |= {t.format(held=h, seen=s): None for h in held for s in seen for t in TWO_OBJECT}
        return finalize(state, candidates)
