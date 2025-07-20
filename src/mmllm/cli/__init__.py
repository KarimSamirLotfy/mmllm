"""CLI package for mmllm benchmarking commands."""

from .benchmark_command import BenchmarkCommand, add_benchmark_command

__all__ = [
    'BenchmarkCommand',
    'add_benchmark_command'
]
