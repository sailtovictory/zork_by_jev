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

from zork_agent import memory, thinker
from zork_agent.errands import forced_action, repeatable
from zork_agent.candidates import TemplateProposer
from zork_agent.frotz import Frotz
from zork_agent.policy import JevPolicy, RandomPolicy
from zork_agent.proposer import ReaderProposer
from zork_agent.reader import LOCAL_MODEL, HaikuReader, LocalReader
from zork_agent.state import DEATH, WorldState
from zork_agent.vocab import load_dictionary

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MEMORY = ROOT / "memory.json"
REPLAY_DELAY = 2.5
# Every game opens the same way, so the leaflet's welcome text is always on screen.
OPENING = ["open mailbox", "read leaflet"]
SAVES = RUNS / "saves"
SAVE_INTERVAL = 15  # turns between checkpoints when nothing else prompts one
MAX_RESTORES = 12  # per game; after that a death stands
MAX_RESTORES_PER_SAVE = 3  # dying this often from one checkpoint means the checkpoint itself is doomed

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
    if record.get("event"):  # a restore after death: not a turn the player took
        print(f"{YELLOW}{BOLD}  ** {record['event']} **{RESET}")
        print(f"\n{CYAN}[step {record['step']}] {record['room']} | score {record['score']} | moves {record['moves']}{RESET}", flush=True)
        return
    if record.get("scripted"):
        print(f"{DIM}  (fixed opening move){RESET}")
    elif record.get("forced"):
        print(f"{DIM}  ({record['forced']}){RESET}")
    elif not record.get("top"):
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


def players(proposer: str, policy: str, model: str | None = None) -> str:
    """Who is playing, in plain words: "Jev and Gemma"."""
    chooser = "Jev" if policy == "jev" else "random choice"
    if proposer == "templates":
        return chooser
    reader = "Claude Haiku" if proposer == "haiku" else (model or LOCAL_MODEL).split(":")[0].rstrip("0123456789.-").capitalize()
    return f"{chooser} and {reader}"


def wait_for_enter(who: str) -> None:
    """Show the title line and hold until Enter is pressed, then clear the screen."""
    input(f"{BOLD}An autonomous agent powered by {who} is about to play Zork I. Press Enter to continue...{RESET}")
    print("\033[2J\033[H", end="", flush=True)


def replay(target: str, delay: float, wait: bool) -> None:
    path = best_log() if target == "best" else Path(target)
    records = read_log(path)
    if wait:
        # Older logs do not record who played; their file names still carry the proposer and policy.
        name = path.name
        proposer = next((kind for kind in ("haiku", "templates") if f"-{kind}-" in name), "local")
        wait_for_enter(records[0].get("players") or players(proposer, "random" if "-random" in name else "jev"))
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
    parser.add_argument("--lesson-writer", choices=["auto", "haiku", "same"], default="auto",
                        help="who writes the end-of-game lessons: Claude Haiku, or the same model that reads the game; "
                             "auto uses Haiku when ANTHROPIC_API_KEY is set")
    parser.add_argument("--until-death", action="store_true",
                        help="end the game when the player dies (or at --steps); turns off restoring saves")
    parser.add_argument("--no-saves", action="store_true", help="do not save the game and restore it after a death")
    parser.add_argument("--hours", type=float, default=None,
                        help="keep starting new games, each learning from the last, until this much time has passed")
    parser.add_argument("--wait", action="store_true", help="wait for Enter before the first turn, for screen recording")
    parser.add_argument("--replay", metavar="LOG", help="replay a saved run ('best' picks the highest score); no API calls")
    parser.add_argument("--dfrotz", type=Path, default=ROOT / "bin" / "dfrotz.exe")
    parser.add_argument("--story", type=Path, default=ROOT / "games" / "zork1.z3")
    args = parser.parse_args()

    os.system("")  # enables ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # notes may contain characters cp1252 lacks
    if args.replay:
        replay(args.replay, REPLAY_DELAY if args.delay is None else args.delay, args.wait)
        return

    load_env(ROOT / ".env")
    reader = {"local": lambda: LocalReader(args.local_model), "haiku": HaikuReader, "templates": lambda: None}[args.proposer]()
    use_haiku = args.lesson_writer == "haiku" or (args.lesson_writer == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
    writer = HaikuReader() if use_haiku and not isinstance(reader, HaikuReader) else reader
    if args.learn:
        for log in args.learn:
            lessons = memory.reflect(MEMORY, read_log(Path(log)), writer)
            print(f"{log}: memory now holds {len(lessons)} lessons")
        return
    rng = random.Random(args.seed)
    if args.policy == "jev":
        policy = JevPolicy(rng, args.model, args.greedy, args.temperature)
    else:
        policy = RandomPolicy(rng)
    proposer = ReaderProposer(reader) if reader else TemplateProposer()

    if args.hours is None:
        play(args, reader, writer, policy, proposer, args.seed)
        return

    # Training: game after game, each starting from the memory the last one left.
    deadline = time.time() + args.hours * 3600
    number = 0
    while time.time() < deadline:
        number += 1
        try:
            result = play(args, reader, writer, policy, proposer, args.seed + number - 1, deadline)
        except Exception as error:  # a dropped connection should not end an unattended run
            print(f"Game {number} failed: {error!r}; retrying in 30 seconds", flush=True)
            time.sleep(30)
            continue
        ending = "died" if result["died"] else "reached the turn or time limit"
        ending += f", {result['deaths']} deaths, {result['restores']} restores"
        recalled = result["recalled"]
        print(f"=== Game {number}: started with {recalled['lessons']} lessons and {recalled['rooms_mapped']} rooms mapped; "
              f"{result['turns']} turns, peak score {result['peak']}, final {result['score']}, "
              f"{result['rooms']} rooms, {ending} ===", flush=True)


def play(args: argparse.Namespace, reader, writer, policy, proposer, seed: int, deadline: float | None = None) -> dict:
    """Play one game, save its log, update memory, and return a summary."""
    use_memory = not args.no_memory
    reflecting = use_memory and writer is not None
    RUNS.mkdir(exist_ok=True)
    log_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.proposer}-{args.policy}-seed{seed}.jsonl"
    SAVES.mkdir(parents=True, exist_ok=True)
    game = Frotz(args.dfrotz, args.story, seed=seed, saves=SAVES)
    saving = not args.no_saves and not args.until_death
    checkpoint: dict | None = None  # {"file", "step", "snapshot", "restores"}
    restores = 0
    state = WorldState(load_dictionary(args.story))
    state.begin(game.read())
    game.send("verbose")  # full room descriptions on every visit, so same-named rooms can be told apart
    if use_memory:
        memory.recall(MEMORY, state)
    recalled = {"lessons": len(state.lessons), "rooms_mapped": len(state.map), "dead_ends": len(state.blocked)}
    records: list[dict] = []
    state.set_inventory(game.send("inventory"))

    who = players(args.proposer, args.policy, getattr(reader, "name", None))
    if args.wait:
        if isinstance(reader, LocalReader):  # load the model now so the first turn is not a long pause on camera
            print(f"{DIM}Loading {reader.name}...{RESET}", flush=True)
            reader.parse("Reply with an empty list.", {}, memory.Lessons, max_tokens=64)
        wait_for_enter(who)

    died = False
    try:
        with log_path.open("w", encoding="utf-8") as log:

            def record_turn(record: dict) -> None:
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                show_turn(record)

            record_turn({"step": 0, "room": state.room, "score": state.score, "moves": state.moves, "response": state.description, "players": who, "memory_at_start": recalled})
            def save_here(step: int) -> None:
                nonlocal checkpoint
                name = f"{log_path.stem}-{step}.qzl"
                if game.save(name):
                    if checkpoint:
                        (SAVES / checkpoint["file"]).unlink(missing_ok=True)
                    checkpoint = {"file": name, "step": step, "snapshot": state.snapshot(), "restores": 0}

            for step in range(1, args.steps + 1):
                if deadline and time.time() > deadline:
                    break  # out of time: stop here so the run ends when it was asked to
                notes_before, score_before = list(state.notes), state.score
                options, info = {}, {}
                if step <= len(OPENING):
                    action, info = OPENING[step - 1], {"scripted": True}
                elif forced := forced_action(state) or (thinker.should_think(state) and (writer or reader)
                                                        and thinker.think(state, writer or reader)
                                                        and forced_action(state)):
                    action, info = forced[0], {"forced": forced[1]}
                else:
                    options = proposer.propose(state)
                    action, info = policy.choose(state, options)

                safe = not state.dark and state.room not in state.deadly_rooms
                if saving and safe and state.map.get(state.room, {}).get(action) in state.deadly_rooms:
                    save_here(step - 1)  # about to walk into a room that has killed before
                room_before = state.room
                state.update(action, game.send(action))
                if "forced" in info and state.room == room_before and not repeatable(action):
                    state.forced_failures.add((room_before, action))  # done, or did not work: either way, not again
                    if "carrying" in state.last_response:
                        state.overloaded = (room_before, action)
                if info.get("forced", "").startswith("trying an idea"):
                    state.record_experiment(room_before, action, state.last_response, state.score > score_before)
                record = {
                    "step": step,
                    "room": state.room,
                    "score": state.score,
                    "moves": state.moves,
                    "action": action,
                    "response": state.last_response,
                    "candidates": len(options),
                    "notes": [note for note in state.notes if note not in notes_before],
                    **info,
                }
                died = DEATH in state.last_response
                if not died and not state.game_over and state.inventory_may_have_changed(action):
                    state.set_inventory(game.send("inventory"))
                    died = DEATH in state.last_response  # killed during the turn the inventory check took
                    record |= {"response": state.last_response, "room": state.room, "score": state.score}
                record_turn(record)
                time.sleep(args.delay or 0)
                if state.game_over or not game.alive or (died and args.until_death):
                    break
                if died:
                    usable = checkpoint and restores < MAX_RESTORES and checkpoint["restores"] < MAX_RESTORES_PER_SAVE
                    if saving and usable:
                        restores += 1
                        checkpoint["restores"] += 1
                        game.restore(checkpoint["file"])
                        state.roll_back(checkpoint["snapshot"], f"The player died in {room_before} after '{action}' "
                                        f"and the game was restored to step {checkpoint['step']}. Do not repeat that.")
                        record_turn({"step": step, "room": state.room, "score": state.score, "moves": state.moves,
                                     "event": f"Died in {room_before}. Restored the save from step {checkpoint['step']}",
                                     "response": ""})
                        continue
                    state.after_death()
                    state.set_inventory(game.send("inventory"))
                safe = not state.dark and not died and state.room not in state.deadly_rooms
                due = checkpoint is None or state.score > score_before or step - checkpoint["step"] >= SAVE_INTERVAL
                if saving and safe and due:
                    save_here(step)
    finally:
        game.close()
        if checkpoint:
            (SAVES / checkpoint["file"]).unlink(missing_ok=True)
        if use_memory:
            memory.save_world(MEMORY, state)
        if reflecting and len(records) > 1:
            lessons = memory.reflect(MEMORY, records, writer)
            print(f"{YELLOW}Memory updated: {len(lessons)} lessons in {MEMORY.name}{RESET}")
    print(f"Finished: score {state.score}, {len(state.visits)} rooms visited, log at {log_path}")
    return {
        "turns": len(records) - 1,
        "score": state.score,
        "peak": max(record["score"] for record in records),
        "rooms": len(state.visits),
        "died": died,
        "deaths": state.deaths,
        "restores": restores,
        "recalled": recalled,
    }


if __name__ == "__main__":
    main()
