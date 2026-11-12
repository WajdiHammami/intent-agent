"""UserChannel adapters that answer the pipeline's questions automatically."""

from adapters.users.scripted import ScriptedUser
from adapters.users.simulated import SimulatedUser

__all__ = ["ScriptedUser", "SimulatedUser"]
