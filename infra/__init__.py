"""Throughput infrastructure (PLAN.md §4): Ray env/policy workers and a
dynamic batcher for the real GPU path, plus a sequential local_backend for
--backend local. Only infra/batcher.py is importable off the GPU box."""
