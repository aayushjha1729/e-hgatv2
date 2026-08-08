"""The E-HGATv2 surrogate (heterogeneous max-plus GATv2) + XGBoost baseline.

Submodules import Torch and PyTorch-Geometric, which belong to the optional learn extra.
They are imported lazily (import ehgat.surrogate.graph) rather than eagerly here, leaving
the environment, oracle and BRKGA layers Torch-free.
"""
