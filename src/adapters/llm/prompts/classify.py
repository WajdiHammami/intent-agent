"""Stage 1 prompt — pick one of the four service classes."""

from langchain_core.prompts import ChatPromptTemplate

CLASS_DESCRIPTIONS = {
    "TimeCritical": (
        "Real-time closed-loop CONTROL of machines, vehicles or grid equipment, with "
        "cycle or trip times of about 1–5 ms (robots, cranes, actuators, protection "
        "relays, V2X coordination, remote/teleoperated machinery). "
        "Latency ≤1 ms, reliability 99.9999%. Always a dedicated slice."
    ),
    "MissionCritical": (
        "Communications for PEOPLE and OPERATIONS that must stay up: public safety, "
        "emergency and disaster response, command and control, critical-infrastructure "
        "monitoring that triggers safety actions. Availability and TN/NTN fallback "
        "matter most. Latency ≤10 ms, availability 99.999%."
    ),
    "RemoteConnectivity": (
        "General BROADBAND or data access for people or sites (internet, video, "
        "telemetry uplinks, crew welfare) where terrestrial networks are missing — "
        "ships, rural villages, polar stations, aircraft, city fringe. Sized by "
        "throughput per site or vessel. Latency ≤500 ms, availability 99.9%."
    ),
    "MassiveConnectivity": (
        "LARGE NUMBERS OF DEVICES — sensors, meters, trackers, tags, buoys — each "
        "sending small periodic reports, usually battery-powered and latency "
        "tolerant. Up to 10⁶ devices/km². Smart metering, environmental and "
        "industrial sensing, asset tracking, agriculture."
    ),
}

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a 5G/6G service class classifier. "
     "Given a natural-language service intent, choose the single most appropriate class. "
     "Decide by WHAT the service does — control loops, mission-critical operations, "
     "broadband access, or many small devices — not by WHERE it runs: being remote, at "
     "sea, in a disaster area or on satellite does not by itself make a request "
     "RemoteConnectivity. "
     "Respond with a JSON object only — no prose, no markdown fences."),
    ("human",
     'Service intent: "{intent}"\n\n'
     "Available classes:\n{class_list}\n\n"
     'Respond with JSON: {{"class_id": "<ID>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}'),
])


def format_class_list() -> str:
    return "\n".join(f'  - "{cid}": {desc}' for cid, desc in CLASS_DESCRIPTIONS.items())
