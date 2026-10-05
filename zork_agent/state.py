"""What the agent knows about the world, rebuilt from dfrotz output each turn."""

import copy
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
ROOM_BUDGET = 12  # commands that neither move nor score before a room is given up on for now
ROOM_LOG_LENGTH = 20
MAX_PUZZLE_NOTES = 3
MAX_PUZZLE_TRIES = 20
MAX_COMBAT_REPEATS = 8
COMBAT_VERBS = {"attack", "kill", "fight", "hit", "stab", "slay", "strike"}
GAME_OVER = "restart the game from the beginning"
DEATH = "You have died"
DARK = "pitch black"
LIGHT_SOURCES = {"lantern", "lamp"}
WEAPONS = {"sword", "knife", "axe", "stiletto"}
PARTING_VERBS = {"drop", "throw", "give", "put"}
# The game counts "inventory" as a move, which burns lantern battery, so it is only asked when the
# list may have changed: not after plain movement or looking at things, unless the reply hints otherwise.
PASSIVE_VERBS = {"examine", "read", "search", "listen", "smell", "wait"}
INVENTORY_HINTS = ("thief", "stole", "robbed", "lantern", "lamp", "taken", "don't have", DEATH)
# What the game says when an enemy is finished, or was never there.
DEFEAT_PHRASES = ("breathes his last", "carcass", " dies", "is dead", "black fog", "can't see any")
MAX_FIGHT_ROUNDS = 30
CLOSED = re.compile(r"[Tt]he ([a-z ]+?) is closed")
SWEEPING_TAKES = {"all", "everything", "treasure", "treasures", "valuables"}
INVENTORY_INTERVAL = 20  # turns between checks when nothing suggests a change
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up",
            "northeast": "southwest", "southwest": "northeast", "northwest": "southeast", "southeast": "northwest",
            "in": "out", "out": "in"}
UNMARKED = "unmarked"
LESSON_RANGE = 2  # lessons about rooms within this many moves are shown
# What a snapshot keeps even when the game is rolled back: knowledge of the world, not of the moment.
KNOWLEDGE = ("map", "signatures", "dark_rooms", "deadly_rooms", "treasures", "trophy", "ambiguous", "fatal", "lessons",
             "puzzles", "thought_about", "lit_rooms", "enemies")
PRONOUNS = {"me", "myself", "self", "you", "it", "them", "all", "everyt"}
INVENTORY_LINE = re.compile(r"(You are carrying|An? |Some |The )")


def base_name(room: str) -> str:
    """The room name without the number or marker that tells look-alikes apart."""
    return re.split(r" [(\[]", room)[0]


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
    # rooms the game described as pitch black, where moving without light is fatal
    dark_rooms: set[str] = field(default_factory=set)
    dark: bool = False  # the player is in the dark right now
    # rooms seen clearly without a lit lantern: they have their own light
    lit_rooms: set[str] = field(default_factory=set)
    # room -> the creature that both blocks the way out and can be fought there
    enemies: dict[str, str] = field(default_factory=dict)
    defeated: set[str] = field(default_factory=set)  # enemies killed or gone, this game
    # commands the harness has lined up for a room: (room, command, why)
    pending: list[tuple[str, str, str]] = field(default_factory=list)
    light_failed: bool = False  # the lantern would not come on (dead battery)
    # rooms where something other than darkness killed the player
    deadly_rooms: set[str] = field(default_factory=set)
    turns_since_inventory: int = 0
    came_from: str = ""  # the room the player was in before this one
    # objects whose taking raised the score
    treasures: set[str] = field(default_factory=set)
    # where treasures score when deposited: {"room": ..., "container": ...}
    trophy: dict[str, str] = field(default_factory=dict)
    # room names shared by rooms that also look alike (the maze); told apart by items dropped in them
    ambiguous: set[str] = field(default_factory=set)
    markers: dict[str, str] = field(default_factory=dict)  # dropped item -> the room it marks
    # (room, action) pairs that killed the player outside a fight
    fatal: set[tuple[str, str]] = field(default_factory=set)
    # (room, action) pairs the harness chose itself and that did not work
    forced_failures: set[tuple[str, str]] = field(default_factory=set)
    # room -> {"notes": what seems stuck, "seen": times noticed, "tried": {command: reply}, "solved": [commands]}
    puzzles: dict[str, dict] = field(default_factory=dict)
    room_log: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # room -> recent (command, reply)
    experiments: list[tuple[str, str, str]] = field(default_factory=list)  # queued (room, command, hypothesis)
    thought_about: set[str] = field(default_factory=set)  # rooms already stopped and thought about this game
    stalled: int = 0  # turns since the score rose or a room was entered for the first time this game
    explore_attempts: dict[str, int] = field(default_factory=dict)  # room -> times the harness set out for it
    idle: dict[str, int] = field(default_factory=dict)  # room -> commands since anything moved or scored there
    failed_takes: dict[tuple[str, str], int] = field(default_factory=dict)  # (room, action) -> takes that got nothing
    deposited: set[str] = field(default_factory=set)  # treasures put in the trophy case this game
    # a (room, action) the harness tried that failed because too much was being carried
    overloaded: tuple[str, str] | None = None
    last_edge: tuple[str, str] | None = None  # the (room, action) that led to the current room
    deaths: int = 0
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

    @property
    def has_light_source(self) -> bool:
        return bool(LIGHT_SOURCES & set(self.inventory_nouns))

    @property
    def armed(self) -> bool:
        return bool(WEAPONS & set(self.inventory_nouns))

    def inventory_may_have_changed(self, action: str) -> bool:
        """False after plain movement or looking at things, when the reply gives no hint of a change."""
        self.turns_since_inventory += 1
        verb = action.split(" ")[0]
        passive = action in self.map.get(self.came_from, {}) or action in DIRECTIONS or verb in PASSIVE_VERBS | COMBAT_VERBS
        hinted = any(hint.lower() in self.last_response.lower() for hint in INVENTORY_HINTS)
        return not passive or hinted or self.turns_since_inventory >= INVENTORY_INTERVAL

    @property
    def lit(self) -> bool:
        return "providing light" in self.inventory_text

    def _key(self, action: str) -> tuple[str, str, str]:
        return (self.room, self.inventory_text, action)

    def _about_carried_items(self, action: str) -> bool:
        """True for reading or examining something in hand, which gives the same answer in every room."""
        nouns = self.nouns(action)
        return action.split(" ")[0] in INFO_VERBS and bool(nouns) and set(nouns) <= set(self.inventory_nouns)

    def _worn_out(self, action: str) -> bool:
        """True once a command that is not an exit has been repeated in this room as often as allowed."""
        verb = action.split(" ")[0]
        if verb == "take":  # successful takes are never capped, so a drop/take cycle ends with the item in hand
            return self.failed_takes.get((self.room, action), 0) >= MAX_REPEATS
        limit = MAX_COMBAT_REPEATS if verb in COMBAT_VERBS else MAX_REPEATS
        return self.uses.get((self.room, action), 0) >= limit and action not in self.map.get(self.room, {})

    @property
    def treasure_words(self) -> set[str]:
        """Every word for a treasure: its known name plus the other nouns on its inventory line ("bag of coins")."""
        words = set(self.treasures)
        for line in self.inventory_text.splitlines():
            nouns = set(self.nouns(line))
            if nouns & self.treasures:
                words |= nouns
        return words

    def reckless(self, action: str) -> bool:
        """True for commands that throw away what keeps the player alive or what the game is played for."""
        verb, nouns = action.split(" ")[0], set(self.nouns(action))
        if action.startswith("turn off") or verb in ("extinguish", "douse"):
            return bool(nouns & LIGHT_SOURCES)
        if verb not in PARTING_VERBS:
            return False
        into_trophy = verb == "put" and self.trophy.get("container") in nouns
        return bool(
            nouns & (LIGHT_SOURCES | {"light"})
            or (nouns & self.treasure_words and not into_trophy)
            or (nouns & WEAPONS and self.room in self.deadly_rooms)
        )

    def _learn_enemy(self, room: str) -> None:
        """A creature is an enemy to fight when it has been attacked here and has also blocked the way out."""
        log = self.room_log.get(base_name(room), [])
        targets = {words[1] for command, _ in log if len(words := command.split(" ")) > 1 and words[0] in COMBAT_VERBS}
        blocked = " ".join(reply.lower() for command, reply in log if command in DIRECTIONS)
        for target in targets:
            if target in blocked:
                self.enemies[base_name(room)] = target

    def enemy_here(self) -> str | None:
        """The enemy to fight in this room, if it is still present."""
        enemy = self.enemies.get(base_name(self.room))
        seen = f"{self.description}\n{self.last_response}".lower()
        return enemy if enemy and enemy not in self.defeated and enemy in seen else None

    @property
    def in_a_fight(self) -> bool:
        return self.room in self.deadly_rooms or any(a.split(" ")[0] in COMBAT_VERBS for a, _ in self.history[-3:])

    def was_tried(self, action: str) -> bool:
        return (
            self.reckless(action) or
            self._key(action) in self.tried
            or action in self.learned
            or (self.room, action) in self.blocked
            or (self.room, action) in self.fatal
            or (action.startswith("take ") and bool(self.deposited & set(self.nouns(action))))
            or (action.startswith("take ") and bool(self.deposited) and self.room == self.trophy.get("room")
                and bool(SWEEPING_TAKES & set(action.split(" "))))
            or self._worn_out(action)
        )

    def tried_here(self) -> list[str]:
        here = {action for room, inventory, action in self.tried if (room, inventory) == self._key("")[:2]}
        here |= {action for room, action in self.blocked | self.fatal if room == self.room}
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

    def note_puzzle(self, description: str) -> None:
        """Record something in this room that could not be explained or got past."""
        puzzle = self.puzzles.setdefault(base_name(self.room), {"notes": [], "seen": 0, "tried": {}, "solved": []})
        puzzle["seen"] += 1
        if description not in puzzle["notes"]:
            puzzle["notes"] = [*puzzle["notes"], description][-MAX_PUZZLE_NOTES:]

    def record_experiment(self, room: str, command: str, reply: str, scored: bool) -> None:
        puzzle = self.puzzles.setdefault(base_name(room), {"notes": [], "seen": 0, "tried": {}, "solved": []})
        puzzle["tried"][command] = reply[:160]
        puzzle["tried"] = dict(list(puzzle["tried"].items())[-MAX_PUZZLE_TRIES:])
        if scored:
            puzzle["solved"].append(command)

    @property
    def room_exhausted(self) -> bool:
        return self.idle.get(self.room, 0) >= ROOM_BUDGET

    def add_notes(self, notes: list[str]) -> None:
        self.notes = [*self.notes, *(note for note in notes if note not in self.notes)][-MAX_NOTES:]

    def begin(self, turn: Turn) -> None:
        self._apply(turn)
        self.description = turn.text
        self.visits[self.room] = 1
        self.lit_rooms.add(self.room)

    def update(self, action: str, turn: Turn) -> None:
        previous_room, key, previous_score = self.room, self._key(action), self.score
        if self._about_carried_items(action):
            self.learned.add(action)
        self.uses[(previous_room, action)] = self.uses.get((previous_room, action), 0) + 1
        self._apply(turn)
        self.last_response = turn.text
        self.history = [*self.history, (action, turn.text)][-HISTORY_LENGTH:]
        log = self.room_log.setdefault(base_name(previous_room), [])
        log.append((action, turn.text[:300]))
        del log[:-ROOM_LOG_LENGTH]
        # In the maze every room looks alike, so a fresh room heading also counts as a move.
        moved = self.room != previous_room or turn.text.splitlines()[:1] == [turn.room]
        if moved and not action.startswith("look"):
            if DEATH not in turn.text:  # being carried off by death is not an exit
                self.map.setdefault(previous_room, {})[action] = self.room
            if self.room == previous_room and DEATH not in turn.text:
                # Moved, yet the room is indistinguishable from the last one: rooms of this name need markers.
                self.ambiguous.add(turn.room)
                self.room = self._identify(turn.room, turn.text)
                self.map[previous_room][action] = self.room
            self.visits[self.room] = self.visits.get(self.room, 0) + 1
            self.description = turn.text
            self.came_from = previous_room
            if self.visits[self.room] == 1:
                self.stalled = -1
            self.last_edge = (previous_room, action)
            self.dark = False
        else:
            self.tried.add(key)
            closed = CLOSED.search(turn.text)
            if closed and action in self.map.get(previous_room, {}) and not self.pending:
                self.pending = [(previous_room, f"open {closed.group(1)}", f"the {closed.group(1)} is closed"),
                                (previous_room, action, "trying that way again")]
            self.idle[self.room] = 0 if self.score > previous_score else self.idle.get(self.room, 0) + 1
            if action in DIRECTIONS:
                self.blocked.add((previous_room, action))
        self._see_darkness(turn.text)
        if action.split(" ")[0] in COMBAT_VERBS or (action in DIRECTIONS and not moved):
            self._learn_enemy(previous_room)
        enemy = self.enemies.get(base_name(previous_room))
        if enemy and enemy in action and any(phrase in turn.text.lower() for phrase in DEFEAT_PHRASES):
            self.defeated.add(enemy)
        if action == "turn on lantern" and "now on" not in turn.text and "already on" not in turn.text:
            self.light_failed = True
        self.stalled = 0 if self.score > previous_score else self.stalled + 1
        verb = action.split(" ")[0]
        if verb == "take" and "Taken" not in turn.text:
            self.failed_takes[(previous_room, action)] = self.failed_takes.get((previous_room, action), 0) + 1
        if DEATH in turn.text:
            self.deaths += 1
            if "grue" not in turn.text:
                self.deadly_rooms.add(previous_room)
            if verb not in COMBAT_VERBS:  # a lost fight may be won next time; a fatal step never is
                self.fatal.add((previous_room, action))
        elif self.score > previous_score and verb == "take":
            self.treasures |= set(self.nouns(action))
        elif self.score > previous_score and verb == "put" and len(self.nouns(action)) > 1:
            self.trophy = {"room": previous_room, "container": self.nouns(action)[-1]}
            self.deposited.add(self.nouns(action)[0])
        elif verb == "drop" and "Dropped" in turn.text and self.room.endswith(f"[{UNMARKED}]") and self.nouns(action):
            self._mark_room(self.nouns(action)[0])

    def _mark_room(self, item: str) -> None:
        """An item dropped in a look-alike room becomes that room's name."""
        marked = f"{base_name(self.room)} [{item}]"
        self.markers[item] = marked
        if self.last_edge and self.map.get(self.last_edge[0], {}).get(self.last_edge[1]) == self.room:
            self.map[self.last_edge[0]][self.last_edge[1]] = marked
        self.visits[marked] = 1
        self.room = marked

    def after_death(self) -> None:
        """The player restarts elsewhere with belongings scattered: forget what only held for the old life."""
        carried = ", ".join(self.inventory_nouns) or "nothing"
        self.notes = [f"The player died and restarted in {self.room}. Carried before dying: {carried}. "
                      "Those items are no longer held; notes made before the death were discarded."]
        self.tried.clear()
        self.learned.clear()
        self.dark = False
        self.came_from = ""
        self.last_edge = None
        self.inventory_text = ""

    def snapshot(self) -> dict:
        """Everything about the moment, to go back to if the game is restored to this point."""
        return copy.deepcopy({name: value for name, value in vars(self).items() if name != "dictionary"})

    def roll_back(self, snapshot: dict, note: str) -> None:
        """Return to a snapshot, keeping what has been learned about the world since it was taken."""
        learned = {name: getattr(self, name) for name in KNOWLEDGE}
        deaths = self.deaths
        for name, value in copy.deepcopy(snapshot).items():
            setattr(self, name, value)
        for name, value in learned.items():
            setattr(self, name, value)
        self.deaths = deaths
        self.add_notes([note])

    def _see_darkness(self, text: str) -> None:
        if DARK in text and DEATH not in text:
            self.dark = True
            self.dark_rooms.add(self.room)
        elif text.splitlines()[:1] == [base_name(self.room)] and not self.lit and DEATH not in text:
            self.lit_rooms.add(self.room)  # described in full with no lantern burning
        elif "is now on" in text:
            self.dark = False

    def set_inventory(self, turn: Turn) -> None:
        room = self.room
        self._apply(turn)
        self.turns_since_inventory = 0
        if DEATH in turn.text:  # asking takes a game turn, and something used it to kill the player
            self.last_response = f"{self.last_response}\n{turn.text}"
            self.deaths += 1
            if "grue" not in turn.text:
                self.deadly_rooms.add(room)
        # Ambient messages (a song bird, the thief) can trail the listing; keep only the listing.
        lines = []
        for line in turn.text.splitlines():
            if not INVENTORY_LINE.match(line):
                break
            lines.append(line)
        self.inventory_text = "\n".join(lines)
        self.deposited -= set(self.inventory_nouns)

    def _identify(self, name: str, text: str) -> str:
        """Name the room, adding a number when a different room has already used this name."""
        lines = text.splitlines()
        if name in self.ambiguous:
            if lines[:1] != [name]:  # no description printed: stay put
                return self.room if base_name(self.room) == name else f"{name} [{UNMARKED}]"
            seen = self.nouns("\n".join(lines[2:]))
            marker = next((item for item in self.markers if item in seen), UNMARKED)
            return f"{name} [{marker}]"
        if lines[:1] == [name] and len(lines) > 1:
            signature = lines[1].split(". ")[0]
            variants = self.signatures.setdefault(name, [])
            if signature not in variants:
                variants.append(signature)
            index = variants.index(signature)
            return name if index == 0 else f"{name} ({index + 1})"
        # No description printed (dark, or not a move): stay put if the name still matches.
        return self.room if base_name(self.room) == name else name

    def _apply(self, turn: Turn) -> None:
        if turn.room is not None:
            self.room, self.score, self.moves = self._identify(turn.room, turn.text), turn.score, turn.moves
        if GAME_OVER in turn.text:
            self.game_over = True

    def relevant_lessons(self) -> list[str]:
        """Lessons about this room, rooms a couple of moves away, or things being carried; and general ones."""
        near = {self.room} | {room for room, route in self.routes().items() if len(route) <= LESSON_RANGE}
        near = {base_name(room).lower() for room in near}
        known = {base_name(room).lower() for room in set(self.map) | set(self.visits)}
        carried = set(self.inventory_nouns)
        kept = []
        for lesson in self.lessons:
            text = lesson.lower()
            mentioned = {room for room in known if room in text}
            if not mentioned or mentioned & near or any(item in text for item in carried):
                kept.append(lesson)
        return kept

    def summary(self) -> dict:
        """The state handed to the decision model."""
        return {
            "lessons_from_past_attempts": self.relevant_lessons(),
            "goal": "Play Zork I. Explore, collect treasures, put them in the trophy case, maximise score, stay alive.",
            "room": self.room,
            "room_description": self.description,
            "last_response": self.last_response,
            "inventory": self.inventory_text,
            "score": self.score,
            "moves": self.moves,
            "known_exits_here": self.map.get(self.room, {}),
            "known_routes_from_here": self.routes(),
            "dark_rooms_needing_a_lit_lantern": sorted(self.dark_rooms),
            "rooms_where_the_player_was_killed_enter_only_with_a_weapon": sorted(self.deadly_rooms),
            "this_game_starts_from_scratch": "Nothing carries over from earlier games. Every treasure is back in "
                                             "its place and must be taken and deposited again this game.",
            "treasures_worth_taking_when_seen": sorted(self.treasures),
            "treasures_deposited_this_game": sorted(self.deposited),
            "where_treasures_score": self.trophy,
            "rooms_visited_this_game": self.visits,
            "notes": self.notes,
            "recent_turns": [{"command": command, "response": response} for command, response in self.history],
        }
