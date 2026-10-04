"""Things the harness does itself, without asking the models: jobs with one right next step."""

from zork_agent.candidates import guarded_exits
from zork_agent.state import LIGHT_SOURCES, UNMARKED, WEAPONS, WorldState, base_name


def _usable(state: WorldState, action: str) -> bool:
    return (state.room, action) not in state.forced_failures and (state.room, action) not in state.fatal


def _mark_maze_room(state: WorldState) -> tuple[str, str] | None:
    """In an unmarked look-alike room, drop something spare so the room can be recognised again."""
    if not state.room.endswith(f"[{UNMARKED}]") or state.dark:
        return None
    keep = LIGHT_SOURCES | WEAPONS | state.treasures | {"light"}
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
        if item not in LIGHT_SOURCES | state.treasures | {"light"} and _usable(state, f"drop {item}"):
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
        steps = [f"open {container}", *(f"put {item} in {container}" for item in carried)]
        return next(((step, reason) for step in steps if _usable(state, step)), None)
    route = state.routes().get(room)
    if not route or not _usable(state, route[0]) or route[0] in guarded_exits(state)[0]:
        return None
    return route[0], reason


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
    return _mark_maze_room(state) or _lighten_load(state) or _deliver_treasure(state) or _run_experiment(state)
