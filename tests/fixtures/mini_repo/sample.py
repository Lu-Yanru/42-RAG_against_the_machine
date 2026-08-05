"""Sample module used as a loader/chunker test fixture."""

import os

FP8_MIN = -448.0
FP8_MAX = 448.0


def helper(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Greeter:
    """Says hello."""

    DEFAULT_NAME = "world"

    def greet(self, name: str = DEFAULT_NAME) -> str:
        return f"Hello, {name}! cwd={os.getcwd()}"
