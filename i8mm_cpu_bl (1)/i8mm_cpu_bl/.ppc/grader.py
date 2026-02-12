#!/usr/bin/env python3

from ppcgrader.cli import cli
import ppci8mm

if __name__ == "__main__":
    cli(
        ppci8mm.Config(code='i8mm_cpu_bl',
                       gpu=False,
                       openmp=False,
                       vnni=False,
                       turing=False))
