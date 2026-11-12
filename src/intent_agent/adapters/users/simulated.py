"""SimulatedUser — an LLM answers questions in the role of a scripted persona."""

from __future__ import annotations

from intent_agent.core.ports import History, UserSimulationModel
from intent_agent.models.dialogue import Question


class SimulatedUser:
    def __init__(self, model: UserSimulationModel, scenario: str) -> None:
        self.model = model
        self.scenario = scenario

    def ask(self, question: Question, history: History) -> str:
        return self.model.simulate_user(self.scenario, history, question.text)
