"""Things the harness does itself, without asking the models: jobs with one right next step."""

from zork_agent.candidates import guarded_exits
from zork_agent.state import LIGHT_SOURCES, MAX_FIGHT_ROUNDS, UNMARKED, WEAPONS, WorldState, base_name


LIGHT, DOUSE = "turn on lantern", "turn off lantern"


def repeatable(action: str) -> bool:
    """Forced commands that are meant to be used more than once in a room."""
    return action in (LIGHT, DOUSE) or action.startswith(("kill ", "put ", "open "))


def _fight(state: WorldState) -> tuple[str, str] | None:
    """Armed, with an enemy that blocks the way: keep attacking until it falls, so the room is safe afterwards."""
    enemy = state.enemy_here()
    weapon = next((item for item in ("sword", "knife", "axe", "stiletto") if item in state.inventory_nouns), None)
    if not enemy or not weapon or state.dark:
        return None
    attack = f"kill {enemy} with {weapon}"
    if state.uses.get((state.room, attack), 0) >= MAX_FIGHT_ROUNDS:
        return None
    return attack, f"finishing the fight with the {enemy}"


def _next_to_the_dark(state: WorldState, room: str) -> bool:
    return any(destination in state.dark_rooms for destination in state.map.get(room, {}).values())


def _manage_lantern(state: WorldState) -> tuple[str, str] | None:
    """The battery is finite: burn it only where it is needed.

    Off once the player is two rooms clear of anything dark, in rooms known to have their own light;
    on again before stepping next to a dark room. A room never seen before is handled by the darkness
    rule, which lights the lantern as soon as the game says it is pitch black.
    """
    if not state.has_light_source or state.dark or state.light_failed:
        return None
    here_dark_adjacent = _next_to_the_dark(state, state.room)
    if not state.lit and here_dark_adjacent:
        return LIGHT, "a dark room is next door"
    clear = state.room in state.lit_rooms and state.came_from in state.lit_rooms
    if state.lit and clear and not here_dark_adjacent and not _next_to_the_dark(state, state.came_from):
        return DOUSE, "saving the lantern battery: this room has its own light"
    return None


def _usable(state: WorldState, action: str) -> bool:
    return (state.room, action) not in state.forced_failures and (state.room, action) not in state.fatal and not state.reckless(action)


def _mark_maze_room(state: WorldState) -> tuple[str, str] | None:
    """In an unmarked look-alike room, drop something spare so the room can be recognised again."""
    if not state.room.endswith(f"[{UNMARKED}]") or state.dark:
        return None
    keep = LIGHT_SOURCES | WEAPONS | state.treasure_words | {"light"}
    for item in state.inventory_nouns:
        if item not in keep and item not in state.markers and _usable(state, f"drop {item}"):
            return f"drop {item}", f"marking this {base_name(state.room)} room with the {item} so it can be told apart"
    return None


def _lighten_load(state: WorldState) -> tuple[str, str] | None:
    """A step on an errand failed because too much is carried: put down everything but light and treasure, then retry."""
    if not state.overloaded or state.overloaded[0] != state.room:
        return None
    reason = "carrying too much for the way ahead"
    for item in state.inventory_nouns:
        if item not in LIGHT_SOURCES | state.treasure_words | {"light"} and _usable(state, f"drop {item}"):
            return f"drop {item}", reason
    action = state.overloaded[1]
    state.overloaded = None
    state.forced_failures.discard((state.room, action))
    return action, reason


def _deliver_treasure(state: WorldState) -> tuple[str, str] | None:
    """Carrying a treasure with a known way to the trophy case: walk there and put it in."""
    carried = [item for item in state.inventory_nouns if item in state.treasures]
    if not carried or not state.trophy or state.dark:
        return None
    room, container = state.trophy["room"], state.trophy["container"]
    reason = f"taking the {carried[0]} to the {container} in the {room}"
    if state.room == room:
        # Counted, not remembered as done: a treasure that comes back out of the case has to go in again.
        steps = [(f"open {container}", 1), *((f"put {item} in {container}", 3) for item in carried)]
        return next(((step, reason) for step, limit in steps if state.uses.get((room, step), 0) < limit), None)
    route = state.routes().get(room)
    if not route or not _usable(state, route[0]) or route[0] in guarded_exits(state)[0]:
        return None
    return route[0], reason


def _run_pending(state: WorldState) -> tuple[str, str] | None:
    """Carry out what the harness lined up for this room, such as opening a closed way and going through."""
    while state.pending and not state.dark:
        room, command, reason = state.pending[0]
        if room != state.room:
            state.pending.clear()  # the player has moved on
            return None
        state.pending.pop(0)
        guarded = command in guarded_exits(state)[0]  # never forced into the dark unlit or into danger unarmed
        if state.uses.get((room, command), 0) < 4 and not state.reckless(command) and not guarded:
            return command, reason
    return None


def _run_experiment(state: WorldState) -> tuple[str, str] | None:
    """Try the next idea from the last time the agent stopped to think about this room."""
    while state.experiments and not state.dark:
        room, command, hypothesis = state.experiments[0]
        if room != base_name(state.room):
            return None  # wait until the player is back in that room
        state.experiments.pop(0)
        if _usable(state, command):
            return command, f"trying an idea: {hypothesis[:110]}"
    return None


def forced_action(state: WorldState) -> tuple[str, str] | None:
    """The command to send without consulting the models, and why; None when the models should decide."""
    return (_fight(state) or _run_pending(state) or _manage_lantern(state) or _mark_maze_room(state) or _lighten_load(state)
            or _deliver_treasure(state) or _run_experiment(state))
