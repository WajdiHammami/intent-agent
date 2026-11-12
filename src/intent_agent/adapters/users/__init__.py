"""UserChannel adapters that answer the pipeline's questions automatically."""

from intent_agent.adapters.users.scripted import ScriptedUser
from intent_agent.adapters.users.simulated import SimulatedUser

__all__ = ["ScriptedUser", "SimulatedUser"]
