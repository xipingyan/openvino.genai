#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Convert a HuggingFace tokenizer to OpenVINO IR format.
# Dev-only tool — not included in release builds.

import argparse
import sys
from pathlib import Path

from openvino import save_model
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer


def convert(
    model_id: str,
    output_dir: str,
    with_detokenizer: bool = True,
    trust_remote_code: bool = False,
    padding_side: str | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    if padding_side:
        tokenizer.padding_side = padding_side
        tokenizer.truncation_side = padding_side

    print(f"Converting tokenizer (with_detokenizer={with_detokenizer}) ...")
    try:
        converted = convert_tokenizer(tokenizer, with_detokenizer=with_detokenizer)
    except NotImplementedError:
        if with_detokenizer:
            print("WARNING: detokenizer not supported for this tokenizer, retrying without it.")
            converted = convert_tokenizer(tokenizer, with_detokenizer=False)
            with_detokenizer = False
        else:
            raise

    if not isinstance(converted, tuple):
        converted = (converted,)

    names = ["openvino_tokenizer.xml"]
    if with_detokenizer and len(converted) > 1:
        names.append("openvino_detokenizer.xml")

    for model, name in zip(converted, names):
        dest = output_path / name
        save_model(model, dest)
        print(f"  Saved: {dest}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace tokenizer to OpenVINO IR format.",
    )
    parser.add_argument(
        "model_id",
        help="HuggingFace model ID or local path containing the tokenizer "
             "(e.g. 'Qwen/Qwen3-0.6B', './my_model').",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the converted IR files (default: model_id directory).",
    )
    parser.add_argument(
        "--no-detokenizer",
        action="store_true",
        help="Skip detokenizer conversion.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )
    parser.add_argument(
        "--padding-side",
        choices=["left", "right"],
        default=None,
        help="Override tokenizer padding/truncation side.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.model_id

    convert(
        model_id=args.model_id,
        output_dir=output_dir,
        with_detokenizer=not args.no_detokenizer,
        trust_remote_code=args.trust_remote_code,
        padding_side=args.padding_side,
    )


if __name__ == "__main__":
    main()
