"""A language model reads the game text each turn: it notes what changed and proposes commands for Jev to choose from."""

from pydantic import BaseModel, Field

from zork_agent.candidates import TemplateProposer, finalize, movement, normalize, parseable
from zork_agent.state import WorldState

SYSTEM = """You read the output of the text adventure Zork I for an autonomous player.

A separate decision model picks the next command; your job is to give it good options and keep its notes accurate.

The state includes lessons_from_past_attempts: what earlier games taught. Follow sequences that scored before when they apply from the current room, and avoid what they say failed.

known_routes_from_here gives the exact commands, in order, to reach each known room from the current one; when a lesson says points are scored in another room, propose the first command of the route there.

From the state you are given, return:
- world_changed: true when the latest response changed the room or an object in it (a door or container opened, something was moved, a creature died or left), false when the command was refused, did nothing, or only gave information. Picking up or dropping an item does not count.
- notes: durable facts learned from the latest response that will matter later (a door that is locked, \
an object that turned out to be a container, where an item was left, what killed the player). Return an empty \
list when the latest response taught nothing new. Do not repeat notes already listed.
- commands: 8 to 15 distinct commands worth trying now, most promising first. Use Zork's parser style: \
a verb, then an object, optionally a preposition and a second object ("open mailbox", "put egg in case", \
"turn on lamp", "kill troll with sword"). Refer only to objects that are visible or carried, \
and name them with the words in objects_here and objects_carried: those are the object words in the \
current text that the game's parser knows. A word missing from both lists will not be understood. Include \
movement only for exits the text mentions. Do not propose commands listed under tried_here, and do not \
propose meta commands such as save, restore, quit, restart, score, inventory, look or g (again).

Command forms the parser accepts (checked against this game's own dictionary):
- movement: north, south, east, west, northeast, northwest, southeast, southwest, up, down, in, out, \
enter [place], climb [object], cross [object], jump
- objects: take [item], take all, drop [item], open / close [container or door], examine [object], \
read [item], search [object], move / push / pull / raise / lower [object], touch / rub [object], wave [item]
- containers: put [item] in [container], fill [container] with [liquid], pour [liquid] on [object]
- devices and light: turn on / turn off [item], light [item] with [item], turn [control] with [item], \
wind [item], ring [item], inflate [item] with [item]
- locks and tools: unlock / lock [door] with [key], dig [place] with [tool], cut [object] with [blade], \
break [object] with [item], tie [item] to [object], burn [object] with [item]
- creatures: attack / kill [creature] with [weapon], give [item] to [creature], throw [item] at [target]
- other: eat [item], drink [item], smell [object], listen to [target], pray, shout, wait, say "[word]"
The parser has no "use" verb; name the specific action instead."""


class Reading(BaseModel):
    world_changed: bool = Field(description="True when the latest response changed the room or an object in it")
    notes: list[str] = Field(description="New durable facts from the latest response; empty if none")
    commands: list[str] = Field(description="Candidate commands, most promising first")


class ReaderProposer:
    """One structured-output call per turn to the text-reading model."""

    def __init__(self, reader):
        self.reader = reader
        self.fallback = TemplateProposer()

    @staticmethod
    def _input(state: WorldState) -> dict:
        carried = state.inventory_nouns
        return {
            **state.summary(),
            "objects_here": [noun for noun in state.room_nouns if noun not in carried],
            "objects_carried": carried,
            "tried_here": state.tried_here(),
        }

    def propose(self, state: WorldState) -> dict[str, str | None]:
        reading = self.reader.parse(SYSTEM, self._input(state), Reading, max_tokens=1024)
        if reading is None:  # refusal or truncated output
            return self.fallback.propose(state)
        if reading.world_changed:
            state.world_changed()
        state.add_notes(reading.notes)
        proposed = {normalize(command): "suggested by the text reader" for command in reading.commands}
        # Words the parser does not know would only waste a turn.
        proposed = {command: hint for command, hint in proposed.items() if parseable(state, command)}
        return finalize(state, proposed | {k: v for k, v in movement(state).items() if k not in proposed})
