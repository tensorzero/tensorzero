#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.40",
#     "tokenizers>=0.15",
# ]
# ///
"""Tokenize text with the multilingual-e5-large tokenizer.

Usage:
    ./tokenize_e5.py "your text here"
    echo "your text" | ./tokenize_e5.py
    ./tokenize_e5.py --ids "your text"
"""

import argparse
import json
import sys

from transformers import AutoTokenizer

MODEL = "intfloat/multilingual-e5-large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Text to tokenize (or read from stdin).")
    parser.add_argument("--ids", action="store_true", help="Print token IDs instead of pieces.")
    parser.add_argument("--count", action="store_true", help="Print only the token count.")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    if not text:
        parser.error("no text provided")

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(text, add_special_tokens=True)
    ids = enc["input_ids"]

    if args.count:
        print(len(ids))
    elif args.ids:
        print(json.dumps(ids))
    else:
        pieces = tok.convert_ids_to_tokens(ids)
        print(json.dumps(pieces, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
