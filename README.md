# Zork agent

An autonomous agent that plays Zork I. A language model (a local Gemma by default) reads the game text and proposes commands; [Jev](https://typesafe.ai), a decision model that returns calibrated probabilities instead of text, picks one. A Python harness runs the game, tracks state, and remembers what it learned between games.

```
Zork I (zork1.z3)
   │
dfrotz ── stdout ──► Python harness
   ▲                   ├─ room, description, inventory, score
   │                   ├─ exact map and dead ends
   │                   ├─ lessons from earlier games
   │                   │
   │                   ▼
   │              Reader (local Gemma, or Claude Haiku)
   │                   └─ proposes 8–15 commands, notes what changed
   │                      (after each game, Claude Haiku writes the lessons)
   │                   ▼
   │              Jev ── picks one command
   │                   │
   └──── command ◄─────┘
```

## Status

Early and experimental. The best game so far scored 60 of 350 points, reached by turn 66. The agent gets into the house, puts the egg in the trophy case, lights the lantern, goes through the trap door, carries the painting back up the chimney to the case, gets past the troll and explores as far as the dam. It is now stuck on puzzles, such as the platinum bar in the Loud Room, and has not come close to finishing the game.

## Requirements

- Windows (the setup script fetches a Windows build of dfrotz; on other systems install `dfrotz` yourself and pass `--dfrotz`)
- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- A Jev API key from TypeSafe AI

The default setup uses two more models, each with a different job:

| Model | Job | Needs |
|---|---|---|
| Gemma (`gemma4:12b`), local | Reads the game text every turn and proposes commands | [Ollama](https://ollama.com) running, about 8 GB of download, a 16 GB GPU |
| Claude Haiku | Writes the lessons at the end of each game | An Anthropic API key |

Either can be left out. Without an Anthropic key, Gemma writes the lessons too. Without Ollama, run with `--proposer haiku` and Haiku does both jobs.

## Setup

```
git clone https://github.com/sailtovictory/zork_by_jev.git
cd zork_by_jev
uv venv
uv pip install -r pyproject.toml
uv run --no-project python scripts/fetch_assets.py
ollama pull gemma4:12b
```

`fetch_assets.py` downloads `dfrotz.exe` from the IF Archive and `zork1.z3` from the [historicalsource/zork1](https://github.com/historicalsource/zork1) repository into `bin/` and `games/`. Skip `ollama pull` if you are using Haiku as the reader. Ollama must be running when you start a game with the local reader.

Copy `.env.example` to `.env` and fill in the keys. `.env` is ignored by git.

## Running

```
uv run --no-project python -m zork_agent --greedy --steps 150 --delay 2
```

Every game opens with `open mailbox` and `read leaflet`, so the welcome text is always shown; the models take over from the third turn. Each turn prints Jev's top five options with their probabilities, the command chosen, the game's reply, and any new notes from the reader.

| Option | Meaning |
|---|---|
| `--proposer local\|haiku\|templates` | What reads the game text. `local` (default) uses Ollama, `haiku` uses Claude Haiku, `templates` uses fixed verb templates and needs no model. |
| `--local-model NAME` | Ollama model to use (default `gemma4:12b`). |
| `--policy jev\|random` | What chooses among the candidates. `random` needs no Jev key. |
| `--greedy` | Always take Jev's top choice. Without it, the command is sampled from Jev's probabilities, which plays noticeably worse. |
| `--steps N` | Number of turns (default 200). |
| `--delay SECONDS` | Pause after each turn, for watching. |
| `--wait` | Show a title line and wait for Enter before the first turn, for screen recording. |
| `--seed N` | Seed for the game's own randomness. |
| `--no-memory` | Do not read or update `memory.json`. |
| `--no-saves` | Do not save the game and restore it after a death. |
| `--until-death` | End the game at the first death. Turns saving off. |
| `--hours H` | Training: keep starting new games, each learning from the last, for this long. |
| `--lesson-writer auto\|haiku\|same` | Who writes the end-of-game lessons. `auto` uses Claude Haiku when an Anthropic key is set, otherwise the reader. |

### Training

```
uv run --no-project python -m zork_agent --greedy --steps 400 --hours 3 > runs\training.log
```

Plays game after game with no pauses, stopping when the time is up even if a game is in progress. Each game loads the memory the last one left. One line per finished game starts with `=== Game`.

### Replaying a game

Every game is saved to `runs/` as a JSONL log. Replays make no API calls.

```
uv run --no-project python -m zork_agent --replay best
uv run --no-project python -m zork_agent --replay runs/<file>.jsonl --delay 1
```

`best` picks the saved run with the highest score reached at any point.

### Pointing at an OpenJev server

The Jev client reads `TYPESAFE_BASE_URL`, so a server that implements the same `/v1/systemone` API can stand in for hosted Jev without code changes. This has not been tested.

## How it works

**Candidates.** Jev cannot generate text; it chooses among options it is given. The reader proposes commands each turn, and the harness filters them against the parser's own dictionary, which it reads out of the story file: commands with unknown words, game-control commands (`save`, `quit`) and commands aimed at the player are dropped. Synonyms are merged (`get` → `take`, `go east` → `east`) so they do not split Jev's vote. Known exits and untried compass directions are added.

**State.** The harness reads the room, score and move count from the status line, and asks the game for the inventory when it may have changed (asking costs a game move, which drains the lantern). Rooms that share a name ("Forest") are told apart by their descriptions.

**Avoiding loops.** Commands that failed are not offered again in the same room with the same inventory. A command that is not an exit can be used twice per room per game (eight times for combat). Exits into much-visited rooms give way to fresher ones, and when every exit is well trodden the agent rotates through them.

**Staying alive.** When the game says it is pitch black, the only options offered are lighting the lantern or going back. Rooms found to be dark are closed until the lantern is lit; rooms where something killed the player are closed until a weapon is carried. The game is saved when the score rises, every 15 turns when safe, and before entering a room that has killed before; on death it is restored, keeping what the death taught.

**The lantern.** Its battery is finite, so the harness turns it off once the player is two rooms clear of anything dark, in rooms it has seen clearly without it, and on again beside a dark room. The models are never offered commands that throw away the light, a treasure, or a weapon in a room that has killed before.

**Errands.** Some jobs have one right next step, so the harness does them without asking the models. An item whose pick-up raised the score is a treasure: when one is carried and a route to the trophy case is known, the harness walks there and puts it in. In rooms that cannot be told apart (the maze), it drops a spare item and names the room after it.

**Puzzles.** The reader flags things it cannot explain or get past, and these go into a per-room journal with what has been tried. When a room's puzzle has been noticed twice, the agent stops once to think: Claude Haiku, with reasoning on, is given only that room's text and history and asked for a hypothesis and a few experiments, which the harness then runs and records. It is told to work as a first-time player, but a model that has read the internet may still recall the game, so this is not a clean test of puzzle solving.

**Memory.** `memory.json` carries knowledge from game to game: the puzzle journal, the exact map and dead ends as the harness observed them, dark and deadly rooms, known treasures and where they score, and up to 25 lessons written after each game from the transcript (what scored, what killed the player, what was needed where). Each turn the models see only the lessons about nearby rooms or carried items. Delete the file to start from scratch.

## Comparing readers

`scripts/compare_readers.py` puts each model in the same ten opening situations and checks whether the right next move is among its proposals.

```
uv run --no-project python scripts/compare_readers.py gemma4:12b haiku
```

| Reader | Right move offered | Seconds per turn |
|---|---|---|
| Claude Haiku 4.5 | 10/10 | 1.8 |
| gemma4:12b | 10/10 | 3.7 |
| qwen3.5:9b | 8/10 | 3.5 |
| qwen3:14b | 7/10 | 3.3 |

Local timings are from an RTX 5080.

## Known limits

- The troll still wins some fights (the game is restored when it does), and the thief is not handled.
- Puzzles that need a leap, such as a play on words, are unsolved.
- Maze marking has only been tested on simulated rooms, and there are only as many markers as spare items.
- The harness only returns treasures along routes it has already walked; the trap door shuts behind the player, so the way back from the cellar has to be found first.
- Readers sometimes propose commands for objects in other rooms.
- Lessons written by a small local model can be wrong; Claude Haiku's have been more accurate.

## Layout

| Path | Contents |
|---|---|
| `zork_agent/frotz.py` | Drives dfrotz over pipes |
| `zork_agent/vocab.py` | Reads the parser dictionary from the story file |
| `zork_agent/state.py` | Room, inventory, map, tried commands |
| `zork_agent/candidates.py` | Filtering, synonym merging, loop breaking, template proposer |
| `zork_agent/reader.py` | Haiku and Ollama backends |
| `zork_agent/proposer.py` | Reader prompt and proposal handling |
| `zork_agent/policy.py` | Jev and random policies |
| `zork_agent/errands.py` | Treasure delivery and maze marking |
| `zork_agent/thinker.py` | The stop-and-think step for puzzles |
| `zork_agent/memory.py` | Map, hazards, treasures and lessons across games |
| `zork_agent/__main__.py` | Command line, play loop, saves, training, display, replay |
