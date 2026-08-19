"""Validated data contracts shared across SmartGroceryAI application layers.

Schemas describe grocery data independently of prompts, model providers, and future
interfaces. This keeps validation behavior consistent whether input comes from an
LLM, an API request, a command-line tool, or a test.
"""
