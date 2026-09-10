"""Compatibility entry point for main_incontext.py --unseen-only."""

import dataclasses
import logging

import tyro

if __package__:
    from . import main_incontext
else:
    import main_incontext


@dataclasses.dataclass
class Args(main_incontext.Args):
    unseen_only: bool = True


def eval_libero(args: Args) -> None:
    main_incontext.eval_libero(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
