#!/usr/bin/env python
"""Training and demo entry points for Shape Analyzer."""

from shape_analyzer.train import train
from shape_analyzer.demo import interactive_demo

__all__ = ["train", "interactive_demo"]

if __name__ == "__main__":
    train()
