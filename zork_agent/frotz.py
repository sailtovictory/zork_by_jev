"""Drive dfrotz over pipes: send a command, get back the text and the status line."""

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

STATUS_RE = re.compile(r"^\s*(?P<room>\S.*?)\s{2,}Score:\s*(?P<score>-?\d+)\s+Moves:\s*(?P<moves>\d+)\s*$")


@dataclass
class Turn:
    text: str
    room: str | None  # None when the status line did not change this turn
    score: int | None
    moves: int | None


class Frotz:
    def __init__(self, dfrotz: Path, story: Path, seed: int | None = None, timeout: float = 10.0,
                 saves: Path | None = None):
        args = [str(dfrotz.resolve()), "-p", "-m", "-w", "250", "-h", "999"]
        if seed is not None:
            args += ["-s", str(seed)]
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [*args, str(story.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=saves,  # dfrotz mangles backslashes in save paths, so saves go by bare name in its own folder
        )
        self._chunks: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while chunk := self.proc.stdout.read1(4096):
            self._chunks.put(chunk)
        self._chunks.put(b"")

    def read(self) -> Turn:
        """Collect output until dfrotz is waiting at its '>' prompt (or has exited)."""
        buffer = b""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                chunk = self._chunks.get(timeout=0.05)
            except queue.Empty:
                if buffer.rstrip(b" ").endswith(b">"):
                    break
                continue
            if not chunk:
                break
            buffer += chunk
        return self._parse(buffer.decode("latin-1").replace("\r", ""))

    def send(self, command: str) -> Turn:
        self.proc.stdin.write(command.encode("latin-1", "replace") + b"\n")
        self.proc.stdin.flush()
        return self.read()

    def _with_filename(self, command: str, name: str) -> Turn:
        """Run save or restore, answering the game's request for a file name."""
        self.proc.stdin.write(command.encode() + b"\n")
        self.proc.stdin.flush()
        asked, deadline = b"", time.monotonic() + self.timeout
        while b"filename" not in asked and time.monotonic() < deadline:
            try:
                asked += self._chunks.get(timeout=0.05)
            except queue.Empty:
                pass
        return self.send(name)

    def save(self, name: str) -> bool:
        """Save the game under a bare file name that does not exist yet."""
        return "Ok." in self._with_filename("save", name).text

    def restore(self, name: str) -> Turn:
        return self._with_filename("restore", name)

    @staticmethod
    def _parse(raw: str) -> Turn:
        raw = raw.rstrip()
        if raw.endswith(">"):
            raw = raw[:-1]
        room = score = moves = None
        lines = []
        for line in raw.split("\n"):
            status = STATUS_RE.match(line)
            if status:
                room, score, moves = status["room"], int(status["score"]), int(status["moves"])
            else:
                lines.append(line.strip())
        return Turn("\n".join(lines).strip(), room, score, moves)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.alive:
            self.proc.kill()
