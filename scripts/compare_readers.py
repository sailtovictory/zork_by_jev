"""Compare text-reading models on the same Zork situations.

For each situation the game is driven to a known point, the model proposes commands, and we check
whether the command a competent player would type next is among them.

uv run --no-project python scripts/compare_readers.py qwen3:14b gemma4:12b
uv run --no-project python scripts/compare_readers.py haiku
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zork_agent.__main__ import load_env  # noqa: E402
from zork_agent.frotz import Frotz  # noqa: E402
from zork_agent.memory import Lessons  # noqa: E402
from zork_agent.proposer import ReaderProposer  # noqa: E402
from zork_agent.reader import HaikuReader, LocalReader  # noqa: E402
from zork_agent.state import WorldState  # noqa: E402
from zork_agent.vocab import load_dictionary  # noqa: E402

STORY = ROOT / "games" / "zork1.z3"

# (commands that set up the situation, any one of these counts as the right next move)
SITUATIONS = [
    ([], {"open mailbox"}),
    (["open mailbox"], {"take leaflet", "take all"}),
    (["north", "east"], {"open window"}),
    (["north", "east", "open window"], {"enter window", "in", "west", "enter house", "climb window"}),
    (["north", "east", "open window", "in"], {"take sack", "take bottle", "take all"}),
    (["north", "east", "open window", "in", "west"], {"take lantern", "take lamp", "take all"}),
    (["north", "east", "open window", "in", "west", "take lamp", "take sword"], {"move rug", "pull rug", "push rug"}),
    (["north", "east", "open window", "in", "west", "take lamp", "take sword", "move rug"], {"open trap door", "open door", "open trapdoor"}),
    (["north", "east", "open window", "in", "west", "take lamp", "take sword", "move rug", "open trap door", "down"],
     {"turn on lamp", "turn on lantern"}),
    (["north", "north", "up"], {"take egg", "take nest", "take all"}),
]


def situation(setup: list[str]) -> tuple[Frotz, WorldState]:
    game = Frotz(ROOT / "bin" / "dfrotz.exe", STORY, seed=0)
    state = WorldState(load_dictionary(STORY))
    state.begin(game.read())
    game.send("verbose")
    state.set_inventory(game.send("inventory"))
    for command in setup:
        state.update(command, game.send(command))
        state.set_inventory(game.send("inventory"))
    return game, state


def main() -> None:
    load_env(ROOT / ".env")
    for name in sys.argv[1:]:
        reader = HaikuReader() if name == "haiku" else LocalReader(name)
        proposer = ReaderProposer(reader)
        reader.parse("Reply with an empty list.", {}, Lessons, 64)  # load the model before timing
        hits, kept, seconds = 0, 0, 0.0
        for setup, wanted in SITUATIONS:
            game, state = situation(setup)
            started = time.time()
            options = proposer.propose(state)
            seconds += time.time() - started
            game.close()
            suggested = [command for command, hint in options.items() if hint == "suggested by the text reader"]
            hit = bool(wanted & set(suggested))
            hits += hit
            kept += len(suggested)
            print(f"  {'ok  ' if hit else 'MISS'} {state.room:<16} wanted {sorted(wanted)[0]!r:<18} got {suggested[:6]}")
        print(f"{name}: {hits}/{len(SITUATIONS)} right move offered, {kept / len(SITUATIONS):.1f} usable commands per turn, "
              f"{seconds / len(SITUATIONS):.1f}s per turn\n")


if __name__ == "__main__":
    main()
