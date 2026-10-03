"""Run the agent, or replay a saved run.

python -m zork_agent [--proposer local|haiku|templates] [--policy jev|random] [--steps N]
python -m zork_agent --replay best|<log file> [--delay SECONDS]
"""

import argparse
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

from zork_agent import memory
from zork_agent.candidates import TemplateProposer
from zork_agent.frotz import Frotz
from zork_agent.policy import JevPolicy, RandomPolicy
from zork_agent.proposer import ReaderProposer
from zork_agent.reader import HaikuReader, LocalReader
from zork_agent.state import WorldState
from zork_agent.vocab import load_dictionary

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MEMORY = ROOT / "memory.json"
REPLAY_DELAY = 2.5

BOLD, DIM, GREEN, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[33m", "\033[0m"
WIDTH = 100


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and value and not key.startswith("#"):
            os.environ.setdefault(key, value.strip("\"'"))


def show_turn(record: dict) -> None:
    """Print one turn so a person can follow along: what was considered, what was typed, what the game said."""
    if record["step"] == 0:  # the opening text, before any command
        print(f"{record['response']}\n", flush=True)
        return
    print(f"{DIM}{'-' * WIDTH}{RESET}")
    for name, probability in record.get("top", {}).items():
        bar = "#" * round(probability * 30)
        marker, colour = (">", GREEN) if name == record["action"] else (" ", DIM)
        print(f"{colour}  {marker} {name:<28} {probability:5.2f} {bar}{RESET}")
    if not record.get("top"):
        print(f"{DIM}  ({record['candidates']} options){RESET}")
    print(f"\n{BOLD}{GREEN}> {record['action']}{RESET}\n")
    for paragraph in record["response"].splitlines():
        print(textwrap.fill(paragraph, WIDTH))
    for note in record.get("notes", []):
        print(f"{YELLOW}  note: {note}{RESET}")
    status = f"[step {record['step']}] {record['room']} | score {record['score']} | moves {record['moves']}"
    print(f"\n{CYAN}{status}{RESET}", flush=True)


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_log() -> Path:
    """The saved run with the highest score reached at any point; ties go to the most rooms seen."""

    def rank(path: Path) -> tuple[int, int]:
        records = read_log(path)
        return max((r["score"] for r in records), default=0), len({r["room"] for r in records})

    logs = [path for path in RUNS.glob("*.jsonl") if path.stat().st_size]
    if not logs:
        raise SystemExit(f"No saved runs in {RUNS}")
    return max(logs, key=rank)


def replay(target: str, delay: float) -> None:
    path = best_log() if target == "best" else Path(target)
    records = read_log(path)
    print(f"{DIM}Replaying {path.name}: {len(records)} turns, top score {max(r['score'] for r in records)}{RESET}\n")
    for record in records:
        show_turn(record)
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["jev", "random"], default="jev")
    parser.add_argument("--proposer", choices=["local", "haiku", "templates"], default="local",
                        help="what reads the game text: a local Ollama model, Claude Haiku, or fixed templates")
    parser.add_argument("--local-model", default=None, help="Ollama model name (default: OLLAMA_MODEL or gemma4:12b)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="Jev model alias (default: jev-latest)")
    parser.add_argument("--greedy", action="store_true", help="always take Jev's top choice instead of sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delay", type=float, default=None, help="seconds to pause after each turn, for watching")
    parser.add_argument("--no-memory", action="store_true", help="neither read nor update lessons from past games")
    parser.add_argument("--learn", metavar="LOG", nargs="+", help="fold saved runs into memory.json and exit")
    parser.add_argument("--replay", metavar="LOG", help="replay a saved run ('best' picks the highest score); no API calls")
    parser.add_argument("--dfrotz", type=Path, default=ROOT / "bin" / "dfrotz.exe")
    parser.add_argument("--story", type=Path, default=ROOT / "games" / "zork1.z3")
    args = parser.parse_args()

    os.system("")  # enables ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # notes may contain characters cp1252 lacks
    if args.replay:
        replay(args.replay, REPLAY_DELAY if args.delay is None else args.delay)
        return

    load_env(ROOT / ".env")
    reader = {"local": lambda: LocalReader(args.local_model), "haiku": HaikuReader, "templates": lambda: None}[args.proposer]()
    if args.learn:
        for log in args.learn:
            lessons = memory.reflect(MEMORY, read_log(Path(log)), reader)
            print(f"{log}: memory now holds {len(lessons)} lessons")
        return
    use_memory = not args.no_memory
    reflecting = use_memory and reader is not None
    rng = random.Random(args.seed)
    if args.policy == "jev":
        policy = JevPolicy(rng, args.model, args.greedy, args.temperature)
    else:
        policy = RandomPolicy(rng)
    proposer = ReaderProposer(reader) if reader else TemplateProposer()

    RUNS.mkdir(exist_ok=True)
    log_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.proposer}-{args.policy}-seed{args.seed}.jsonl"
    game = Frotz(args.dfrotz, args.story, seed=args.seed)
    state = WorldState(load_dictionary(args.story))
    state.begin(game.read())
    game.send("verbose")  # full room descriptions on every visit, so same-named rooms can be told apart
    if use_memory:
        memory.recall(MEMORY, state)
    records: list[dict] = []
    state.set_inventory(game.send("inventory"))

    try:
        with log_path.open("w", encoding="utf-8") as log:

            def record_turn(record: dict) -> None:
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                show_turn(record)

            record_turn({"step": 0, "room": state.room, "score": state.score, "moves": state.moves, "response": state.description})
            for step in range(1, args.steps + 1):
                notes_before = list(state.notes)
                options = proposer.propose(state)
                action, info = policy.choose(state, options)
                state.update(action, game.send(action))
                record_turn({
                    "step": step,
                    "room": state.room,
                    "score": state.score,
                    "moves": state.moves,
                    "action": action,
                    "response": state.last_response,
                    "candidates": len(options),
                    "notes": [note for note in state.notes if note not in notes_before],
                    **info,
                })
                time.sleep(args.delay or 0)
                if state.game_over or not game.alive:
                    break
                state.set_inventory(game.send("inventory"))
    finally:
        game.close()
        if use_memory:
            memory.save_world(MEMORY, state)
        if reflecting and len(records) > 1:
            lessons = memory.reflect(MEMORY, records, reader)
            print(f"{YELLOW}Memory updated: {len(lessons)} lessons in {MEMORY.name}{RESET}")
    print(f"Finished: score {state.score}, {len(state.visits)} rooms visited, log at {log_path}")


if __name__ == "__main__":
    main()
