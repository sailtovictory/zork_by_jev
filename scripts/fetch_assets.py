"""Download dfrotz.exe (Windows) and the Zork I story file into bin/ and games/."""

import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DFROTZ_ZIP = "https://www.ifarchive.org/if-archive/infocom/interpreters/frotz/dfrotz.zip"
ZORK1 = "https://github.com/historicalsource/zork1/raw/master/COMPILED/zork1.z3"


def fetch(url: str) -> bytes:
    # The IF Archive refuses urllib's default user agent.
    request = urllib.request.Request(url, headers={"User-Agent": "zork-agent-setup"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def main() -> None:
    dfrotz = ROOT / "bin" / "dfrotz.exe"
    if not dfrotz.exists():
        dfrotz.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(DFROTZ_ZIP))) as archive:
            dfrotz.write_bytes(archive.read("dfrotz.exe"))
        print(f"wrote {dfrotz}")
    story = ROOT / "games" / "zork1.z3"
    if not story.exists():
        story.parent.mkdir(exist_ok=True)
        story.write_bytes(fetch(ZORK1))
        print(f"wrote {story}")


if __name__ == "__main__":
    main()
