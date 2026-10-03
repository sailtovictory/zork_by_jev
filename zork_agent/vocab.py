"""Read the parser's dictionary out of a Z-machine v3 story file.

Infocom's dictionary tags each word with its part of speech, so the nouns the game
understands can be recognised in room text instead of guessed at.
"""

from pathlib import Path

A0 = "abcdefghijklmnopqrstuvwxyz"
A2 = " \n0123456789.,!?_#'\"/\\-:()"
NOUN, VERB, ADJECTIVE, DIRECTION = 0x80, 0x40, 0x20, 0x10
WORD_LENGTH = 6  # v3 dictionary words are truncated to six characters


def _decode(data: bytes) -> str:
    zchars = []
    for i in range(0, len(data), 2):
        word = data[i] << 8 | data[i + 1]
        zchars += [word >> 10 & 31, word >> 5 & 31, word & 31]
    text, shift = "", False
    for z in zchars:
        if z == 5 and not shift:
            shift = True
        elif z >= 6:
            text += (A2 if shift else A0)[z - 6]
            shift = False
    return text


def load_dictionary(story: Path) -> dict[str, int]:
    """Map each dictionary word (at most six characters) to its part-of-speech flags."""
    data = story.read_bytes()
    address = data[8] << 8 | data[9]
    address += 1 + data[address]  # skip the word-separator list
    entry_length = data[address]
    count = data[address + 1] << 8 | data[address + 2]
    address += 3
    words = {}
    for i in range(count):
        entry = data[address + i * entry_length : address + (i + 1) * entry_length]
        words[_decode(entry[:4])] = entry[4]
    return words
