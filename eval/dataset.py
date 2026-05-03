"""
TMP 3.0 — Evaluation Dataset

Labeled intents used for quantitative evaluation.

Each EvalSample has:
  - intent            : natural-language service request
  - scenario          : deployment context for the LLM user simulator
  - expected_class    : ground-truth service class
  - expected_slots    : {field: value} subset that MUST appear in the output
  - expected_conflicts: list of conflict descriptions expected to be detected
  - request_type      : seed | paraphrased | ambiguous | contradictory
  - scenario_label    : Urban | RuralRemote | Disaster | Maritime
  - gold_deployment_mode: TN_Only | NTN_Only | Hybrid  (empty = not checked)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    id: str
    intent: str
    scenario: str
    expected_class: str
    expected_slots: dict = field(default_factory=dict)    # must match (exact or subset)
    expected_conflicts: list[str] = field(default_factory=list)  # substrings to match
    request_type: str = "seed"         # seed | paraphrased | ambiguous | contradictory
    scenario_label: str = ""           # Urban | RuralRemote | Disaster | Maritime
    gold_deployment_mode: str = ""     # TN_Only | NTN_Only | Hybrid  (empty = not checked)


# ── Group A — TimeCritical (5 samples) ───────────────────────────────────────

TC_SAMPLES = [
    EvalSample(
        id="TC-01",
        intent=(
            "We need real-time synchronized control for 60 robotic welding arms "
            "on a single automotive assembly floor. Control loop must run at 1 ms. "
            "Dedicated slice required, budget is Premium."
        ),
        scenario=(
            "You are the OT engineer at an automotive factory. "
            "Application type: RoboticControl. Cycle time: 1 ms. "
            "Geographic scope: Campus. Cost: Premium. Sector: Manufacturing."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "RoboticControl",
            "isolation_level": "Dedicated",
            "priority_level": "Critical",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="TC-02",
        intent=(
            "Our smart grid substation needs protection relay automation. "
            "Trip time must be under 4 ms per IEC 61850 class P1. "
            "The substation covers one campus. Balanced cost is fine."
        ),
        scenario=(
            "You are a grid automation engineer. Application: GridProtection. "
            "Trip time: 3 ms. Geographic scope: Campus. Cost: Balanced. Sector: Energy."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "GridProtection",
            "isolation_level": "Dedicated",
            "priority_level": "Critical",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="TC-03",
        intent=(
            "We are deploying a V2X system for platooning trucks on a highway. "
            "Latency must be sub-millisecond for safe coordination. "
            "Dedicated 5G slice on a regional highway corridor, Premium budget."
        ),
        scenario=(
            "You are a transportation systems engineer. Application: V2X. "
            "V2X scenario: Platooning. Cycle time: 0.8 ms. "
            "Geographic scope: Urban. Cost: Premium. Sector: Automotive."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "V2X",
            "priority_level": "Critical",
            "isolation_level": "Dedicated",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="TC-04",
        intent=(
            "Remote process control for a chemical plant requires deterministic "
            "1 ms cycle time for actuator commands. Single facility on one campus. "
            "We can afford Balanced cost."
        ),
        scenario=(
            "You are a process control engineer at a chemical plant. "
            "Application: RemoteProcessControl. Cycle time: 1 ms. "
            "Geographic scope: Campus. Cost: Balanced. Sector: Industry."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "RemoteProcessControl",
            "isolation_level": "Dedicated",
            "priority_level": "Critical",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="TC-05",
        intent=(
            "Factory automation for a semiconductor fab: robotic arms must "
            "coordinate within 0.5 ms. Clock sync accuracy of 1 microsecond required. "
            "Campus deployment, Premium slice."
        ),
        scenario=(
            "You are the network engineer at a semiconductor fab. "
            "Application: RoboticControl. Cycle time: 0.5 ms. "
            "Sync accuracy: 1 µs. Geographic scope: Campus. Cost: Premium. Sector: Manufacturing."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "RoboticControl",
            "priority_level": "Critical",
            "isolation_level": "Dedicated",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
]


# ── Group B — MissionCritical (5 samples) ────────────────────────────────────

MC_SAMPLES = [
    EvalSample(
        id="MC-01",
        intent=(
            "We are setting up a nationwide emergency communications network for "
            "police, fire, and ambulance. It must survive terrestrial infrastructure "
            "failures — LEO satellite fallback required. Premium, dedicated slice."
        ),
        scenario=(
            "You are the CTO of a national public safety agency. "
            "Fallback mode: NTN_Fallback. NTN tier: LEO. "
            "Max failover: 30 seconds. Geographic scope: National. "
            "Isolation: Dedicated. Cost: Premium. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="seed",
        scenario_label="Disaster",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="MC-02",
        intent=(
            "Critical infrastructure protection for a national power grid. "
            "The system must remain operational during disasters. "
            "Hybrid TN/NTN deployment across multiple regions. Balanced cost."
        ),
        scenario=(
            "You are a grid security engineer. Deployment mode: Hybrid. "
            "NTN tier: LEO. Max failover: 60 seconds. "
            "Geographic scope: Regional. Isolation: Dedicated. Cost: Balanced. Sector: Energy."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="seed",
        scenario_label="Disaster",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="MC-03",
        intent=(
            "Urban command and control for city emergency services. "
            "Terrestrial network is sufficient — no satellite needed. "
            "City-wide coverage, dedicated slice, Premium."
        ),
        scenario=(
            "You are the city IT director for emergency services. "
            "Deployment mode: TN_Only. Geographic scope: CityWide. "
            "Isolation: Dedicated. Cost: Premium. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "TN_Only",
            "priority_level": "High",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="MC-04",
        intent=(
            "Oil refinery process monitoring requires mission-critical connectivity "
            "with guaranteed failover to satellite. Coverage spans a large industrial campus. "
            "Latency can be up to 50 ms. Balanced cost."
        ),
        scenario=(
            "You are the IT manager at an oil refinery. "
            "Deployment mode: Hybrid. NTN tier: LEO. Max failover: 10 seconds. "
            "Geographic scope: Campus. Isolation: Shared. Cost: Balanced. Sector: Energy."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="MC-05",
        intent=(
            "Public safety network for a disaster-prone coastal region. "
            "Must use LEO satellite when terrestrial fails. "
            "Failover must complete within 20 seconds. Regional scope, Premium."
        ),
        scenario=(
            "You are the emergency services network planner. "
            "Deployment mode: Hybrid. NTN tier: LEO. Max failover: 20 seconds. "
            "Geographic scope: Regional. Isolation: Dedicated. Cost: Premium. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="seed",
        scenario_label="Disaster",
        gold_deployment_mode="Hybrid",
    ),
]


# ── Group C — RemoteConnectivity (5 samples) ─────────────────────────────────

RC_SAMPLES = [
    EvalSample(
        id="RC-01",
        intent=(
            "150 offshore oil platforms in the North Sea need broadband connectivity. "
            "Minimum 10 Mbps per platform for crew communications and telemetry. "
            "Latency up to 300 ms is acceptable. LEO satellite preferred."
        ),
        scenario=(
            "You are the network director for an offshore oil company. "
            "Geographic scope: Maritime. NTN tier: LEO. Min throughput: 10 Mbps. "
            "Latency budget: 300 ms. Vessel count: 150. "
            "Isolation: Shared. Cost: Balanced. Sector: Energy."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Maritime",
            "priority_level": "Normal",
        },
        request_type="seed",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="RC-02",
        intent=(
            "Rural broadband for 200 remote villages across a mountainous region. "
            "Minimum 5 Mbps per village, latency under 500 ms is fine. "
            "LEO satellite is the only viable option."
        ),
        scenario=(
            "You are a rural connectivity project manager. "
            "Geographic scope: Rural. NTN tier: LEO. Min throughput: 5 Mbps. "
            "Latency budget: 400 ms. Isolation: Shared. Cost: Balanced. Sector: Government."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Rural",
            "priority_level": "Normal",
        },
        request_type="seed",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="RC-03",
        intent=(
            "UAV fleet management for aerial surveying drones over national parks. "
            "Need reliable telemetry at 2 Mbps per drone, latency up to 200 ms. "
            "UAV-HAPS relay preferred for on-demand coverage."
        ),
        scenario=(
            "You are the operations manager for a drone survey company. "
            "Geographic scope: Aerial. NTN tier: UAV_HAPS. Min throughput: 2 Mbps. "
            "Latency budget: 200 ms. Isolation: Shared. Cost: LowCost. Sector: Government."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Aerial",
            "priority_level": "Normal",
        },
        request_type="seed",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="RC-04",
        intent=(
            "Scientific research stations in Antarctica need connectivity for "
            "data uploads and video conferencing. 1 Mbps minimum, latency of "
            "several hundred ms is acceptable. GEO satellite available."
        ),
        scenario=(
            "You are the IT manager for a polar research institute. "
            "Geographic scope: Polar. NTN tier: GEO. Min throughput: 1 Mbps. "
            "Latency budget: 600 ms. Interactive traffic: False. "
            "Isolation: Shared. Cost: Balanced. Sector: Government."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Polar",
            "priority_level": "Normal",
        },
        request_type="seed",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="RC-05",
        intent=(
            "Fishing vessels need connectivity at sea for navigation, weather updates, "
            "and crew welfare internet access. Roughly 300 vessels in the region. "
            "10 Mbps per vessel, latency under 400 ms. LEO satellite."
        ),
        scenario=(
            "You are the fleet manager for a fishing company. "
            "Geographic scope: Maritime. NTN tier: LEO. Min throughput: 10 Mbps. "
            "Latency budget: 400 ms. Vessel count: 300. "
            "Isolation: Shared. Cost: Balanced. Sector: Maritime."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Maritime",
            "priority_level": "Normal",
        },
        request_type="seed",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
]


# ── Group D — MassiveConnectivity (5 samples) ─────────────────────────────────

DC_SAMPLES = [
    EvalSample(
        id="DC-01",
        intent=(
            "Deploying 2 million smart electricity meters across a city. "
            "Each meter reports consumption every 30 minutes. "
            "Battery-powered devices, 10-year lifetime target. 200-byte payloads."
        ),
        scenario=(
            "You are the IoT architect for a utility company. "
            "Device density: 1500/km2. Reporting interval: 1800 s. "
            "Payload: 200 bytes. Battery: yes. Target lifetime: 10 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: Energy."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="DC-02",
        intent=(
            "Agricultural IoT: 500,000 soil moisture and weather sensors "
            "across a national farming region. Sensors report every 15 minutes. "
            "Devices are solar+battery, small payloads of 50 bytes."
        ),
        scenario=(
            "You are the IoT engineer for a smart farming platform. "
            "Device density: 200/km2. Reporting interval: 900 s. "
            "Payload: 50 bytes. Battery: yes. Target lifetime: 5 years. "
            "Geographic scope: Regional. Isolation: Shared. Cost: LowCost. Sector: Industry."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="seed",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="DC-03",
        intent=(
            "Smart city infrastructure: 1 million parking sensors, street lights, "
            "and waste bins reporting status every hour. Very small payloads. "
            "City-wide deployment, cost must be minimal."
        ),
        scenario=(
            "You are the smart city project manager. "
            "Device density: 5000/km2. Reporting interval: 3600 s. "
            "Payload: 20 bytes. Battery: yes. Target lifetime: 7 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: SmartCity."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="DC-04",
        intent=(
            "Industrial factory floor monitoring: 200,000 vibration sensors on "
            "machines reporting every 5 minutes. Payload is 100 bytes per report. "
            "Devices are mains-powered. Campus deployment."
        ),
        scenario=(
            "You are the IIoT engineer at a large manufacturing plant. "
            "Device density: 10000/km2. Reporting interval: 300 s. "
            "Payload: 100 bytes. Battery: no. "
            "Geographic scope: Campus. Isolation: Shared. Cost: Balanced. Sector: Manufacturing."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="DC-05",
        intent=(
            "Water utility: smart water meters for an entire city, reporting flow "
            "every 20 minutes. 800,000 devices, battery-powered, 8-year target. "
            "Messages are 150 bytes. Shared slice is fine."
        ),
        scenario=(
            "You are the network engineer for a water utility. "
            "Device density: 1200/km2. Reporting interval: 1200 s. "
            "Payload: 150 bytes. Battery: yes. Target lifetime: 8 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: SmartCity."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="seed",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
]


# ── Conflict samples (2 samples) ──────────────────────────────────────────────

CONFLICT_SAMPLES = [
    EvalSample(
        id="CF-01",
        intent=(
            "We need a dedicated 5G slice for a critical control system "
            "but want the absolute lowest cost possible."
        ),
        scenario=(
            "You are an engineer who asked for the cheapest option. "
            "When the system flags a conflict between Dedicated slice and LowCost, "
            "you agree to upgrade to Balanced. "
            "Application: RoboticControl. Cycle time: 2 ms. "
            "Geographic scope: Campus. Sector: Manufacturing."
        ),
        expected_class="TimeCritical",
        expected_slots={},
        expected_conflicts=["dedicated", "LowCost"],
        request_type="contradictory",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="CF-02",
        intent=(
            "Mission-critical emergency network with Critical SLA priority "
            "but we insist on LowCost profile to stay within tight budget."
        ),
        scenario=(
            "You are a budget-constrained IT manager. When told Critical+LowCost "
            "is incompatible, you agree to upgrade to Balanced. "
            "Deployment mode: TN_Only. Geographic scope: Urban. "
            "Isolation: Shared. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={},
        expected_conflicts=["Critical", "LowCost"],
        request_type="contradictory",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
]


# ── Paraphrased samples (8 samples) ──────────────────────────────────────────

PAR_SAMPLES = [
    EvalSample(
        id="PAR-TC-01",
        intent=(
            "Deploy a 5G slice for synchronized motion control across 60 industrial "
            "welding robots on an automobile production line. Control cycle must not "
            "exceed 1 ms. Isolated slice, Premium budget."
        ),
        scenario=(
            "You are the OT engineer at an automotive factory. "
            "Application type: RoboticControl. Cycle time: 1 ms. "
            "Geographic scope: Campus. Cost: Premium. Sector: Manufacturing."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "RoboticControl",
            "isolation_level": "Dedicated",
        },
        request_type="paraphrased",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="PAR-TC-02",
        intent=(
            "Highway truck convoy automation needs wireless synchronization below 1 ms "
            "for safe platooning. Dedicated 5G on a regional highway corridor. "
            "Premium budget."
        ),
        scenario=(
            "You are a transportation systems engineer. Application: V2X. "
            "V2X scenario: Platooning. Cycle time: 0.8 ms. "
            "Geographic scope: Urban. Cost: Premium. Sector: Automotive."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "V2X",
            "isolation_level": "Dedicated",
        },
        request_type="paraphrased",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="PAR-MC-01",
        intent=(
            "National emergency services — police, fire, and EMS — need a dedicated "
            "resilient network. Must stay operational if ground infrastructure fails, "
            "using LEO satellite as backup. Premium cost."
        ),
        scenario=(
            "You are the CTO of a national public safety agency. "
            "Deployment mode: Hybrid. NTN tier: LEO. Max failover: 30 seconds. "
            "Geographic scope: National. Isolation: Dedicated. Cost: Premium. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="paraphrased",
        scenario_label="Disaster",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="PAR-MC-02",
        intent=(
            "Mission-critical monitoring at a petroleum refinery must switch to "
            "satellite if the terrestrial link fails. Single-site campus, "
            "moderate cost acceptable."
        ),
        scenario=(
            "You are the IT manager at an oil refinery. "
            "Deployment mode: Hybrid. NTN tier: LEO. Max failover: 10 seconds. "
            "Geographic scope: Campus. Isolation: Shared. Cost: Balanced. Sector: Energy."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
            "priority_level": "High",
        },
        request_type="paraphrased",
        scenario_label="Urban",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="PAR-RC-01",
        intent=(
            "Over 150 oil extraction platforms in the North Sea need broadband for "
            "operations and crew. Each needs at least 10 Mbps, a few hundred "
            "milliseconds latency is fine. Prefer satellite."
        ),
        scenario=(
            "You are the network director for an offshore oil company. "
            "Geographic scope: Maritime. NTN tier: LEO. Min throughput: 10 Mbps. "
            "Latency budget: 300 ms. Vessel count: 150. "
            "Isolation: Shared. Cost: Balanced. Sector: Energy."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Maritime",
            "priority_level": "Normal",
        },
        request_type="paraphrased",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="PAR-RC-02",
        intent=(
            "Bring internet to 200 isolated communities in mountainous terrain. "
            "Minimum 5 Mbps per community, latency under 500 ms. "
            "LEO satellite is the only viable option."
        ),
        scenario=(
            "You are a rural connectivity project manager. "
            "Geographic scope: Rural. NTN tier: LEO. Min throughput: 5 Mbps. "
            "Latency budget: 400 ms. Isolation: Shared. Cost: Balanced. Sector: Government."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Rural",
            "priority_level": "Normal",
        },
        request_type="paraphrased",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="PAR-DC-01",
        intent=(
            "Roll out wireless connectivity for 2 million residential electricity "
            "meters across the city. Each sends a reading every 30 minutes. "
            "Battery-powered, 10-year lifetime. Messages are 200 bytes."
        ),
        scenario=(
            "You are the IoT architect for a utility company. "
            "Device density: 1500/km2. Reporting interval: 1800 s. "
            "Payload: 200 bytes. Battery: yes. Target lifetime: 10 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: Energy."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="paraphrased",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="PAR-DC-02",
        intent=(
            "City-wide IoT for parking sensors, street lighting, and refuse bins — "
            "about 1 million nodes, checking in hourly. Tiny data payloads. "
            "Budget must be minimal."
        ),
        scenario=(
            "You are the smart city project manager. "
            "Device density: 5000/km2. Reporting interval: 3600 s. "
            "Payload: 20 bytes. Battery: yes. Target lifetime: 7 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: SmartCity."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="paraphrased",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
]


# ── Ambiguous samples (8 samples — missing key slot) ─────────────────────────

AMB_SAMPLES = [
    EvalSample(
        id="AMB-TC-01",
        intent=(
            "Industrial robot control for a car manufacturing plant. "
            "Real-time closed-loop control needed. Dedicated 5G slice, Premium."
        ),
        scenario=(
            "You are the OT engineer at an automotive factory. "
            "Application type: RoboticControl. Cycle time: 1 ms. "
            "Geographic scope: Campus. Cost: Premium. Sector: Manufacturing."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "application_type": "RoboticControl",
        },
        request_type="ambiguous",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="AMB-TC-02",
        intent=(
            "Ultra-low latency 5G slice for critical industrial automation. "
            "1 ms cycle time, dedicated slice, campus deployment."
        ),
        scenario=(
            "You are a process control engineer. Application: RemoteProcessControl. "
            "Cycle time: 1 ms. Geographic scope: Campus. Cost: Balanced. Sector: Industry."
        ),
        expected_class="TimeCritical",
        expected_slots={
            "cycle_time_ms": 1.0,
        },
        request_type="ambiguous",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="AMB-MC-01",
        intent=(
            "Highly available communications network for emergency services in a "
            "metropolitan area. Dedicated slice, Premium cost."
        ),
        scenario=(
            "You are the city IT director for emergency services. "
            "Deployment mode: TN_Only. Geographic scope: CityWide. "
            "Isolation: Dedicated. Cost: Premium. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "priority_level": "High",
        },
        request_type="ambiguous",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="AMB-MC-02",
        intent=(
            "Mission-critical public safety network needing satellite backup for "
            "disaster resilience. Dedicated slice, Balanced cost."
        ),
        scenario=(
            "You are the emergency services network planner. "
            "Deployment mode: Hybrid. NTN tier: LEO. Max failover: 30 seconds. "
            "Geographic scope: Regional. Isolation: Dedicated. Cost: Balanced. Sector: Government."
        ),
        expected_class="MissionCritical",
        expected_slots={
            "deployment_mode": "Hybrid",
        },
        request_type="ambiguous",
        scenario_label="Disaster",
        gold_deployment_mode="Hybrid",
    ),
    EvalSample(
        id="AMB-RC-01",
        intent=(
            "Offshore fishing fleet connectivity for 200 vessels at sea. "
            "Need at least 10 Mbps per vessel, 400 ms latency is fine."
        ),
        scenario=(
            "You are the fleet manager for a fishing company. "
            "NTN tier: LEO. Geographic scope: Maritime. Min throughput: 10 Mbps. "
            "Latency budget: 400 ms. Vessel count: 200. "
            "Isolation: Shared. Cost: Balanced. Sector: Maritime."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Maritime",
        },
        request_type="ambiguous",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="AMB-RC-02",
        intent=(
            "Remote connectivity for polar research stations. "
            "GEO satellite available. Latency of several hundred ms is fine."
        ),
        scenario=(
            "You are the IT manager for a polar research institute. "
            "NTN tier: GEO. Min throughput: 1 Mbps. Geographic scope: Polar. "
            "Latency budget: 600 ms. Interactive traffic: False. "
            "Isolation: Shared. Cost: Balanced. Sector: Government."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={
            "geographic_scope": "Polar",
        },
        request_type="ambiguous",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="AMB-DC-01",
        intent=(
            "Smart city IoT for parking and waste bin sensors. "
            "Small payloads, hourly reporting, battery-powered, 7-year target. "
            "Minimal cost."
        ),
        scenario=(
            "You are the smart city project manager. "
            "Device density: 3000/km2. Reporting interval: 3600 s. "
            "Payload: 30 bytes. Battery: yes. Target lifetime: 7 years. "
            "Geographic scope: CityWide. Isolation: Shared. Cost: LowCost. Sector: SmartCity."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="ambiguous",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
    EvalSample(
        id="AMB-DC-02",
        intent=(
            "Agricultural sensor network for 500,000 soil and weather monitoring "
            "devices across a farming region. Payloads of 50 bytes, "
            "battery-powered, 5-year lifetime."
        ),
        scenario=(
            "You are the IoT engineer for a smart farming platform. "
            "Reporting interval: 900 s. Device density: 200/km2. "
            "Payload: 50 bytes. Battery: yes. Target lifetime: 5 years. "
            "Geographic scope: Regional. Isolation: Shared. Cost: LowCost. Sector: Agriculture."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={
            "priority_level": "Low",
        },
        request_type="ambiguous",
        scenario_label="RuralRemote",
        gold_deployment_mode="NTN_Only",
    ),
]


# ── New Contradiction samples (5 samples) ─────────────────────────────────────

CF_NEW_SAMPLES = [
    EvalSample(
        id="CF-03",
        intent=(
            "Deploy terrestrial-only IoT connectivity for maritime buoys and "
            "coastal sea-level monitoring stations. Budget must be minimal."
        ),
        scenario=(
            "You are the IoT engineer for a coastal monitoring authority. "
            "When the system flags TN-only is not viable in Maritime, you agree to "
            "switch to NTN_Only. NTN tier: LEO. Geographic scope: Maritime. "
            "Device density: 50/km2. Reporting interval: 3600 s. "
            "Payload: 50 bytes. Battery: yes. Isolation: Shared. Cost: LowCost. Sector: Government."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={},
        expected_conflicts=["Maritime", "TN_Only"],
        request_type="contradictory",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
    EvalSample(
        id="CF-04",
        intent=(
            "Critical robot control for an offshore platform requires a Hybrid "
            "TN/NTN deployment. The NTN layer should use GEO satellite for "
            "wide-area coverage."
        ),
        scenario=(
            "You are the OT engineer on an offshore oil platform. "
            "When told GEO is incompatible with Critical priority, you agree to "
            "switch to LEO. Application: RoboticControl. Cycle time: 2 ms. "
            "Geographic scope: Maritime. Cost: Premium. Sector: Energy."
        ),
        expected_class="TimeCritical",
        expected_slots={},
        expected_conflicts=["GEO", "Critical"],
        request_type="contradictory",
        scenario_label="Maritime",
        gold_deployment_mode="Hybrid",
    ),
    # CF-05: MissionCritical — GEO satellite + latency-sensitive ops (class-specific conflict)
    EvalSample(
        id="CF-05",
        intent=(
            "We need a mission-critical telemedicine network for remote surgery "
            "assistance in rural hospitals. Real-time video with sub-100 ms latency "
            "is required. GEO satellite preferred for wide national coverage."
        ),
        scenario=(
            "You are the CTO of a telemedicine platform. When the system flags "
            "that GEO RTT (~540 ms) is incompatible with latency-sensitive surgical "
            "operations, you agree to switch to LEO. "
            "Deployment mode: Hybrid. Geographic scope: National. "
            "Max failover: 30 seconds. Isolation: Dedicated. Cost: Premium. Sector: Healthcare."
        ),
        expected_class="MissionCritical",
        expected_slots={},
        expected_conflicts=["GEO", "latency"],
        request_type="contradictory",
        scenario_label="RuralRemote",
        gold_deployment_mode="Hybrid",
    ),
    # CF-06: RemoteConnectivity — GEO satellite + interactive_traffic (class-specific conflict)
    EvalSample(
        id="CF-06",
        intent=(
            "Offshore oil rigs in the Gulf of Mexico need live two-way video "
            "conferencing with onshore engineers and real-time interactive control. "
            "GEO satellite is already installed on all platforms."
        ),
        scenario=(
            "You are the network manager for an offshore energy company. "
            "When told GEO RTT is incompatible with interactive traffic, "
            "you agree to switch to LEO. "
            "NTN tier: GEO. Geographic scope: Maritime. Min throughput: 10 Mbps. "
            "Latency budget: 80 ms. Interactive traffic: True. "
            "Isolation: Shared. Cost: Balanced. Sector: Energy."
        ),
        expected_class="RemoteConnectivity",
        expected_slots={},
        expected_conflicts=["GEO", "interactive"],
        request_type="contradictory",
        scenario_label="Maritime",
        gold_deployment_mode="NTN_Only",
    ),
    # CF-07: MassiveConnectivity — Dedicated slice + Premium cost (class-specific conflict)
    EvalSample(
        id="CF-07",
        intent=(
            "Deploy 2 million smart water meters across the city with a fully "
            "dedicated Premium network slice to guarantee zero interference."
        ),
        scenario=(
            "You are the IoT architect for a water utility. When told a dedicated "
            "Premium slice is economically unviable for massive IoT at this scale, "
            "you agree to switch to Shared isolation and LowCost profile. "
            "Device density: 1200/km2. Reporting interval: 1200 s. "
            "Payload: 150 bytes. Battery: yes. Target lifetime: 8 years. "
            "Geographic scope: CityWide. Sector: SmartCity."
        ),
        expected_class="MassiveConnectivity",
        expected_slots={},
        expected_conflicts=["Dedicated", "Premium"],
        request_type="contradictory",
        scenario_label="Urban",
        gold_deployment_mode="TN_Only",
    ),
]


# ── Full dataset ──────────────────────────────────────────────────────────────

ALL_SAMPLES: list[EvalSample] = (
    TC_SAMPLES
    + MC_SAMPLES
    + RC_SAMPLES
    + DC_SAMPLES
    + CONFLICT_SAMPLES
    + PAR_SAMPLES
    + AMB_SAMPLES
    + CF_NEW_SAMPLES
)
