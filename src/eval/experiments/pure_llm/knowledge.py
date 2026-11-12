"""
The domain knowledge the Pure-LLM baseline gets *as text* in its prompts,
instead of the typed knowledge graph.
"""

CLASS_INFO: dict[str, dict] = {
    "TimeCritical": {
        "description": (
            "Ultra-low latency industrial/vehicular control (≤1 ms, 99.9999% reliability). "
            "Robotics, grid protection, V2X platooning. Always Dedicated slice, Critical priority."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "application_type", "cycle_time_ms",
        ],
        "implied": {"priority_level": "Critical", "isolation_level": "Dedicated"},
    },
    "MissionCritical": {
        "description": (
            "Resilience-first connectivity (≤10 ms, 99.999% availability). "
            "Public safety, emergency response, critical infrastructure. "
            "Mandatory TN/NTN fallback support."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "isolation_level", "cost_profile", "deployment_mode",
        ],
        "conditional_note": (
            "If deployment_mode is Hybrid or NTN_Only, also ask: "
            "ntn_tier (LEO/GEO/UAV_HAPS) and max_failover_sec."
        ),
        "implied": {"priority_level": "High"},
    },
    "RemoteConnectivity": {
        "description": (
            "NTN-primary (satellite/UAV) coverage for areas without terrestrial networks "
            "(≤500 ms, 99.9% availability). Maritime, rural, aerial, polar."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "ntn_tier", "min_throughput_mbps",
        ],
        "conditional_note": (
            "If the service involves voice/video/real-time interaction, "
            "also ask: interactive_traffic (True/False)."
        ),
        "implied": {"priority_level": "Normal"},
    },
    "MassiveConnectivity": {
        "description": (
            "Massive numbers of low-power IoT devices (≥10⁶ dev/km², 1 Kbps/device). "
            "Smart metering, industrial sensors, precision agriculture."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "device_density_per_km2",
        ],
        "conditional_note": (
            "Also useful: reporting_interval_sec, payload_bytes, battery (True/False), "
            "target_lifetime_years."
        ),
        "implied": {"priority_level": "Low"},
    },
}

CONFLICT_RULES = """\
Known incompatible value combinations (flag ALL that apply):
  1. priority_level=Critical  + cost_profile=LowCost
       → Critical SLA cannot be guaranteed at LowCost; use Balanced or Premium.
  2. isolation_level=Dedicated + cost_profile=LowCost
       → Dedicated slices are not offered at LowCost; use Balanced or Premium.
  3. deployment_scenario=Maritime + deployment_mode=TN_Only
       → No terrestrial network in Maritime; use NTN_Only or Hybrid.
  4. ntn_tier=GEO + priority_level=Critical
       → GEO RTT ~540 ms cannot meet Critical latency requirements; use LEO.
  5. deployment_scenario=Disaster + deployment_mode=TN_Only
       → TN infrastructure may be degraded/destroyed in Disaster; use Hybrid or NTN_Only.
  6. ntn_tier=GEO + (interactive_traffic=True OR any latency requirement ≤ 200 ms)
       → GEO RTT ~540 ms is incompatible with interactive or latency-sensitive applications.
  7. isolation_level=Dedicated + cost_profile=Premium + class=MassiveConnectivity
       → Dedicated Premium slices are economically unviable for massive IoT scale."""
