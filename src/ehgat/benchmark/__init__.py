"""Effectiveness benchmark (multi-seed runs, faithfulness, plots).

Submodules import Torch (the surrogate), scipy, matplotlib and the optional viz extra.
They are imported lazily (import ehgat.benchmark.runner) to leave the Torch-free
environment, oracle and BRKGA layers importable on their own.
"""
