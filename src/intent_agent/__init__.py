"""
Intent Agent — knowledge-guided LLM pipeline that turns natural-language 6G
service requests into validated service profiles.

Layers (dependencies point downwards only):

    cli/, eval/        user interfaces and experiments
    bootstrap.py       composition root: wires adapters into the core
    adapters/          LangChain + Ollama model, simulated/scripted users
    core/              ports, knowledge graph, LangGraph pipeline
    models/            plain dataclasses
"""

__version__ = "0.2.0"
