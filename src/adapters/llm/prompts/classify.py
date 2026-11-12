"""Stage 1 prompt — pick one of the four service classes."""

from langchain_core.prompts import ChatPromptTemplate

CLASS_DESCRIPTIONS = {
    "TimeCritical": (
        "Sub-millisecond industrial/vehicular control. "
        "Latency ≤1 ms, reliability 99.9999%. "
        "Robotics, grid protection relays, V2X. Always dedicated slice."
    ),
    "MissionCritical": (
        "Resilience-first connectivity where availability matters most. "
        "Latency ≤10 ms, availability 99.999%. "
        "Public safety, emergency response, critical infrastructure. "
        "Mandatory TN/NTN fallback support."
    ),
    "RemoteConnectivity": (
        "NTN-primary (satellite/UAV) coverage for areas without terrestrial networks. "
        "Latency ≤500 ms, availability 99.9%. "
        "Maritime, rural, aerial, polar, city fringe."
    ),
    "MassiveConnectivity": (
        "Massive numbers of low-power, low-rate IoT devices. "
        "Density ≥10⁶ dev/km², throughput 1 Kbps/device, latency tolerant. "
        "Smart metering, industrial sensors, precision agriculture."
    ),
}

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a 5G/6G service class classifier. "
     "Given a natural-language service intent, choose the single most appropriate class. "
     "Respond with a JSON object only — no prose, no markdown fences."),
    ("human",
     'Service intent: "{intent}"\n\n'
     "Available classes:\n{class_list}\n\n"
     'Respond with JSON: {{"class_id": "<ID>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}'),
])


def format_class_list() -> str:
    return "\n".join(f'  - "{cid}": {desc}' for cid, desc in CLASS_DESCRIPTIONS.items())
