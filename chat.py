#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from os import environ
from typing import Optional
import sys
import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    system_message: str,
    user_message: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # if environ["RANK"] == 0:
    #     sys.stdin = open(0)
    # else:
    #     sys.stdin = None

    dialog = []
    print(f"> system: {system_message}")
    dialog.append({"role": "system", "content": system_message})

    while True:
        print(f"> user: {user_message}")
        dialog.append({"role": "user", "content": user_message})

        result = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]["generation"]
        print(f"> {result['role']}: {result['content']}")
        dialog.append(result)


if __name__ == "__main__":
    fire.Fire(main)
