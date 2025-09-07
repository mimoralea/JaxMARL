#!/usr/bin/env python3
"""
Compatibility shim for scripted behaviors.

This module re-exports the public API from `baselines.behaviors.scripted` to
preserve backward compatibility for existing imports:

    from baselines import scripted_behaviors

Recommended new import path:

    from baselines.behaviors import scripted

"""
from baselines.behaviors.scripted import (
    get_scripted_action,
    list_scripted_behaviors,
    get_scripted_agent,
)

__all__ = [
    "get_scripted_action",
    "list_scripted_behaviors",
    "get_scripted_agent",
]
