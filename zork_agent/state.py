"""What the agent knows about the world, rebuilt from dfrotz output each turn."""

import re
from collections import deque
from dataclasses import dataclass, field

from zork_agent.frotz import Turn
from zork_agent.vocab import DIRECTION, NOUN, WORD_LENGTH

HISTORY_LENGTH = 8
MAX_NOTES = 30
DIRECTIONS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "up", "down", "in", "out"]
INFO_VERBS = {"read", "examine"}
# How often one command may be used in one room per game. Fights take several blows; little else needs repeating.
MAX_REPEATS = 2
MAX_COMBAT_REPEATS = 8
COMBAT_VERBS = {"attack", "kill", "fight", "hit", "stab", "slay", "strike"}
GAME_OVER = "restart the game from the beginning"
DEATH = "You have died"
PRONOUNS = {"me", "myself", "self", "you", "it", "them", "all", "everyt"}
INVENTORY_LINE = re.compile(r"(You are carrying|An? |Some |The )")


@dataclass
class WorldState:
    dictionary: dict[str, int]
    room: str = ""
    description: str = ""
    last_response: str = ""
    inventory_text: str = ""
    score: int = 0
    moves: int = 0
    game_over: bool = False
    history: list[tuple[str, str]] = field(default_factory=list)
    # room -> action -> destination room; exact, taken from the status line after each command
    map: dict[str, dict[str, str]] = field(default_factory=dict)
    # room name -> the distinct opening sentences seen under that name, to tell same-named rooms apart
    signatures: dict[str, list[str]] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    # actions already tried in a given (room, inventory) that did not move the player
    tried: set[tuple[str, str, str]] = field(default_factory=set)
    # (room, direction) pairs that went nowhere
    blocked: set[tuple[str, str]] = field(default_factory=set)
    # (room, action) -> times used this game
    uses: dict[tuple[str, str], int] = field(default_factory=dict)
    # read/examine commands about carried items, already answered once
    learned: set[str] = field(default_factory=set)

    def nouns(self, text: str) -> list[str]:
        """Words in the text that the game's parser knows as objects."""
        found = {}
        for word in re.findall(r"[a-z]+", text.lower()):
            flags = self.dictionary.get(word[:WORD_LENGTH], 0)
            if flags & NOUN and not flags & DIRECTION and word[:WORD_LENGTH] not in PRONOUNS:
                found.setdefault(word[:WORD_LENGTH], word)
        return list(found.values())

    @property
    def room_nouns(self) -> list[str]:
        return self.nouns(f"{self.description}\n{self.last_response}")

    @property
    def inventory_nouns(self) -> list[str]:
        return self.nouns(self.inventory_text)

    def _key(self, action: str) -> tuple[str, str, str]:
        return (self.room, self.inventory_text, action)

    def _about_carried_items(self, action: str) -> bool:
        """True for reading or examining something in hand, which gives the same answer in every room."""
        nouns = self.nouns(action)
        return action.split(" ")[0] in INFO_VERBS and bool(nouns) and set(nouns) <= set(self.inventory_nouns)

    def _worn_out(self, action: str) -> bool:
        """True once a command that is not an exit has been repeated in this room as often as allowed."""
        verb = action.split(" ")[0]
        if verb == "take":  # never capped, so a drop/take cycle ends with the item in hand, not on the floor
            return False
        limit = MAX_COMBAT_REPEATS if verb in COMBAT_VERBS else MAX_REPEATS
        return self.uses.get((self.room, action), 0) >= limit and action not in self.map.get(self.room, {})

    def was_tried(self, action: str) -> bool:
        return (
            self._key(action) in self.tried
            or action in self.learned
            or (self.room, action) in self.blocked
            or self._worn_out(action)
        )

    def tried_here(self) -> list[str]:
        here = {action for room, inventory, action in self.tried if (room, inventory) == self._key("")[:2]}
        here |= {action for room, action in self.blocked if room == self.room}
        here |= {action for room, action in self.uses if room == self.room and self._worn_out(action)}
        return sorted(here | self.learned)

    def world_changed(self) -> None:
        """Something in this room changed (a door opened, a creature died): what failed before may work now."""
        last_action = self.history[-1][0] if self.history else None
        self.tried = {key for key in self.tried if key[0] != self.room or key[2] == last_action}
        self.blocked = {pair for pair in self.blocked if pair[0] != self.room}

    def routes(self) -> dict[str, list[str]]:
        """Shortest known command sequence from the current room to every room reachable on the map."""
        routes: dict[str, list[str]] = {self.room: []}
        queue = deque([self.room])
        while queue:
            room = queue.popleft()
            for action, destination in self.map.get(room, {}).items():
                if destination not in routes:
                    routes[destination] = [*routes[room], action]
                    queue.append(destination)
        del routes[self.room]
        return routes

    def add_notes(self, notes: list[str]) -> None:
        self.notes = [*self.notes, *(note for note in notes if note not in self.notes)][-MAX_NOTES:]

    def begin(self, turn: Turn) -> None:
        self._apply(turn)
        self.description = turn.text
        self.visits[self.room] = 1

    def update(self, action: str, turn: Turn) -> None:
        previous_room, key = self.room, self._key(action)
        if self._about_carried_items(action):
            self.learned.add(action)
        self.uses[(previous_room, action)] = self.uses.get((previous_room, action), 0) + 1
        self._apply(turn)
        self.last_response = turn.text
        self.history = [*self.history, (action, turn.text)][-HISTORY_LENGTH:]
        # In the maze every room looks alike, so a fresh room heading also counts as a move.
        moved = self.room != previous_room or turn.text.splitlines()[:1] == [turn.room]
        if moved and not action.startswith("look"):
            if DEATH not in turn.text:  # being carried off by death is not an exit
                self.map.setdefault(previous_room, {})[action] = self.room
            self.visits[self.room] = self.visits.get(self.room, 0) + 1
            self.description = turn.text
        else:
            self.tried.add(key)
            if action in DIRECTIONS:
                self.blocked.add((previous_room, action))

    def set_inventory(self, turn: Turn) -> None:
        self._apply(turn)
        # Ambient messages (a song bird, the thief) can trail the listing; keep only the listing.
        lines = []
        for line in turn.text.splitlines():
            if not INVENTORY_LINE.match(line):
                break
            lines.append(line)
        self.inventory_text = "\n".join(lines)

    def _identify(self, name: str, text: str) -> str:
        """Name the room, adding a number when a different room has already used this name."""
        lines = text.splitlines()
        if lines[:1] == [name] and len(lines) > 1:
            signature = lines[1].split(". ")[0]
            variants = self.signatures.setdefault(name, [])
            if signature not in variants:
                variants.append(signature)
            index = variants.index(signature)
            return name if index == 0 else f"{name} ({index + 1})"
        # No description printed (dark, or not a move): stay put if the name still matches.
        return self.room if self.room.split(" (")[0] == name else name

    def _apply(self, turn: Turn) -> None:
        if turn.room is not None:
            self.room, self.score, self.moves = self._identify(turn.room, turn.text), turn.score, turn.moves
        if GAME_OVER in turn.text:
            self.game_over = True

    def summary(self) -> dict:
        """The state handed to the decision model."""
        return {
            "lessons_from_past_attempts": self.lessons,
            "goal": "Play Zork I. Explore, collect treasures, put them in the trophy case, maximise score, stay alive.",
            "room": self.room,
            "room_description": self.description,
            "last_response": self.last_response,
            "inventory": self.inventory_text,
            "score": self.score,
            "moves": self.moves,
            "known_exits_here": self.map.get(self.room, {}),
            "known_routes_from_here": self.routes(),
            "rooms_visited_this_game": self.visits,
            "notes": self.notes,
            "recent_turns": [{"command": command, "response": response} for command, response in self.history],
        }
