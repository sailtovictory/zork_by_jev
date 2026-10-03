# Zork agent

An autonomous agent that plays Zork I. A language model reads the game text and proposes commands; [Jev](https://typesafe.ai), a decision model that returns calibrated probabilities instead of text, picks one. A Python harness runs the game, tracks state, and remembers what it learned between games.

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
   │                   ▼
   │              Jev ── picks one command
   │                   │
   └──── command ◄─────┘
```

## Status

Early and experimental. The best game so far scored 39 of 350 points; a typical 100-turn game reaches 35 points and 20–25 rooms. The agent gets into the house, through the trap door and past the troll, and explores the first underground rooms. It does not yet bring treasures back to the trophy case, which is where most of the points are.

## Requirements

- Windows (the setup script fetches a Windows build of dfrotz; on other systems install `dfrotz` yourself and pass `--dfrotz`)
- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- A Jev API key from TypeSafe AI
- For the reader, one of:
  - [Ollama](https://ollama.com) with `gemma4:12b` (about 8 GB; runs on a 16 GB GPU)
  - an Anthropic API key, for Claude Haiku

## Setup

```
uv venv
uv pip install -r pyproject.toml
uv run --no-project python scripts/fetch_assets.py
ollama pull gemma4:12b
```

`fetch_assets.py` downloads `dfrotz.exe` from the IF Archive and `zork1.z3` from the [historicalsource/zork1](https://github.com/historicalsource/zork1) repository into `bin/` and `games/`.

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

**State.** After every command the harness asks the game for the inventory and reads the room, score and move count from the status line. Rooms that share a name ("Forest") are told apart by their descriptions.

**Avoiding loops.** Commands that failed are not offered again in the same room with the same inventory. A command that is not an exit can be used twice per room per game (eight times for combat). Exits into much-visited rooms give way to fresher ones, and when every exit is well trodden the agent rotates through them.

**Memory.** `memory.json` carries two things from game to game: the exact map and dead ends as the harness observed them, and up to 25 lessons the reader writes after each game from the transcript (what scored, what killed the player, what was needed where). Delete the file to start from scratch.

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

- The agent often enters the cellar without the lantern and dies, even though memory holds a lesson telling it not to.
- Dying is not handled: notes about where items were left go stale.
- All maze rooms look alike, so the map cannot tell them apart.
- Asking for the inventory every turn costs a game move, so the lantern battery drains about twice as fast as for a human player.
- Readers sometimes propose commands for objects in other rooms.

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
| `zork_agent/memory.py` | Map and lessons across games |
| `zork_agent/__main__.py` | Command line, display, logging, replay |
