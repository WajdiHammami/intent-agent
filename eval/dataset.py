"""
Evaluation dataset for the knowledge-guided intent-to-profile framework.

Each :class:`EvalSample` carries:
  - intent               : natural-language service request
  - scenario             : deployment context for the LLM user simulator
  - expected_class       : ground-truth service class
  - expected_slots       : {field: value} subset that MUST appear in the output
  - expected_conflicts   : conflict descriptions expected to be detected
  - request_type         : seed | paraphrased | ambiguous | contradictory
  - scenario_label       : Urban | RuralRemote | Disaster | Maritime
  - gold_deployment_mode : TN_Only | NTN_Only | Hybrid  (empty = not checked)

Exposed collections
-------------------
  ALL_SAMPLES           72 samples — the main evaluation set, balanced across
                        the four service classes (18 per class: 9 seed,
                        3 paraphrased, 3 ambiguous, 3 contradictory).
  CONTRADICTORY_SUBSET  40 contradictory-only samples (CF-01..CF-40) spanning
                        all eight conflict-rule types; used for the KG
                        conflict-rule sweep and the Pure-LLM comparison.
  NON_CONTRADICTORY     The 60 well-formed samples inside ALL_SAMPLES.

ALL_SAMPLES contains NON_CONTRADICTORY plus the 12 contradictory samples listed
in ``_SELECTED_CF_IDS``; the remaining 28 contradictory samples live only in
CONTRADICTORY_SUBSET. The union of both sets is 100 unique samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalSample:
    id: str
    intent: str
    scenario: str
    expected_class: str
    expected_slots: dict = field(default_factory=dict)            # subset match
    expected_conflicts: list[str] = field(default_factory=list)   # substring match
    request_type: str = "seed"        # seed | paraphrased | ambiguous | contradictory
    scenario_label: str = ""          # Urban | RuralRemote | Disaster | Maritime
    gold_deployment_mode: str = ""    # TN_Only | NTN_Only | Hybrid  (empty = unchecked)


# ────────────────────────────────────────────────────────────────────────────
# Seed intents — hand-authored, one realistic scenario per vertical (36)
# ────────────────────────────────────────────────────────────────────────────

_SEED_SAMPLES = [
    EvalSample(
        id='EXT-TC-01',
        intent=(
            'Autonomous container crane at a smart port: needs 2 ms control '
            'cycle for safe lift operations. Dedicated Premium slice. Campus '
            'deployment.'
        ),
        scenario=(
            'You are the port automation engineer. Application: RoboticControl. '
            'Cycle time: 2 ms. Geographic scope: Campus. Cost: Premium. Sector: '
            'Transportation.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'isolation_level': 'Dedicated'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-TC-02',
        intent=(
            'Underground mine conveyor belt control: requires deterministic 2 '
            'ms timing to prevent jams. One campus site. Balanced cost.'
        ),
        scenario=(
            'You are the mining systems engineer. Application: '
            'RemoteProcessControl. Cycle time: 2 ms. Geographic scope: Campus. '
            'Cost: Balanced. Sector: Industry.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RemoteProcessControl', 'priority_level': 'Critical'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-TC-03',
        intent=(
            'Aircraft maintenance robotic arm system at an airport hangar. '
            'Precise sub-2 ms synchronized movement. Dedicated slice, Premium.'
        ),
        scenario=(
            'You are the MRO engineer at an airline. Application: '
            'RoboticControl. Cycle time: 1.5 ms. Geographic scope: Campus. '
            'Cost: Premium. Sector: Transportation.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'isolation_level': 'Dedicated'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-TC-04',
        intent=(
            'Nuclear power station reactor control: actuator commands must be '
            'deterministic under 1 ms. Fully dedicated slice required. Premium.'
        ),
        scenario=(
            'You are the nuclear control systems engineer. Application: '
            'RemoteProcessControl. Cycle time: 1 ms. Geographic scope: Campus. '
            'Cost: Premium. Sector: Energy.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RemoteProcessControl', 'isolation_level': 'Dedicated'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='TC-01',
        intent=(
            'We need real-time synchronized control for 60 robotic welding arms '
            'on a single automotive assembly floor. Control loop must run at 1 '
            'ms. Dedicated slice required, budget is Premium.'
        ),
        scenario=(
            'You are the OT engineer at an automotive factory. Application '
            'type: RoboticControl. Cycle time: 1 ms. Geographic scope: Campus. '
            'Cost: Premium. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'isolation_level': 'Dedicated', 'priority_level': 'Critical'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='TC-02',
        intent=(
            'Our smart grid substation needs protection relay automation. Trip '
            'time must be under 4 ms per IEC 61850 class P1. The substation '
            'covers one campus. Balanced cost is fine.'
        ),
        scenario=(
            'You are a grid automation engineer. Application: GridProtection. '
            'Trip time: 3 ms. Geographic scope: Campus. Cost: Balanced. Sector: '
            'Energy.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'GridProtection', 'isolation_level': 'Dedicated', 'priority_level': 'Critical'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='TC-03',
        intent=(
            'We are deploying a V2X system for platooning trucks on a highway. '
            'Latency must be sub-millisecond for safe coordination. Dedicated '
            '5G slice on a regional highway corridor, Premium budget.'
        ),
        scenario=(
            'You are a transportation systems engineer. Application: V2X. V2X '
            'scenario: Platooning. Cycle time: 0.8 ms. Geographic scope: Urban. '
            'Cost: Premium. Sector: Automotive.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'V2X', 'priority_level': 'Critical', 'isolation_level': 'Dedicated'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='TC-04',
        intent=(
            'Remote process control for a chemical plant requires deterministic '
            '1 ms cycle time for actuator commands. Single facility on one '
            'campus. We can afford Balanced cost.'
        ),
        scenario=(
            'You are a process control engineer at a chemical plant. '
            'Application: RemoteProcessControl. Cycle time: 1 ms. Geographic '
            'scope: Campus. Cost: Balanced. Sector: Industry.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RemoteProcessControl', 'isolation_level': 'Dedicated', 'priority_level': 'Critical'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='TC-05',
        intent=(
            'Factory automation for a semiconductor fab: robotic arms must '
            'coordinate within 0.5 ms. Clock sync accuracy of 1 microsecond '
            'required. Campus deployment, Premium slice.'
        ),
        scenario=(
            'You are the network engineer at a semiconductor fab. Application: '
            'RoboticControl. Cycle time: 0.5 ms. Sync accuracy: 1 µs. '
            'Geographic scope: Campus. Cost: Premium. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'priority_level': 'Critical', 'isolation_level': 'Dedicated'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-MC-01',
        intent=(
            'Dam structural monitoring and emergency alert system. Must remain '
            'operational if terrestrial network fails — LEO satellite fallback. '
            'Regional coverage. Premium, dedicated.'
        ),
        scenario=(
            'You are the dam safety engineer. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 30 s. Geographic scope: Regional. '
            'Isolation: Dedicated. Cost: Premium. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='EXT-MC-02',
        intent=(
            'Airport operations center communications: air traffic control, '
            'ground crew, and emergency response. Must stay up if TN fails. '
            'Citywide coverage, dedicated, Balanced cost.'
        ),
        scenario=(
            'You are the airport IT director. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 10 s. Geographic scope: CityWide. '
            'Isolation: Dedicated. Cost: Balanced. Sector: Transportation.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='EXT-MC-03',
        intent=(
            'Border security command network for a 500 km land border. Must '
            'operate in remote areas with no terrestrial coverage. LEO '
            'satellite primary, Premium, dedicated.'
        ),
        scenario=(
            'You are the border security network engineer. Deployment mode: '
            'NTN_Only. NTN tier: LEO. Max failover: 60 s. Geographic scope: '
            'National. Isolation: Dedicated. Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'NTN_Only', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-MC-04',
        intent=(
            'Water treatment plant control and monitoring needs 99.999% '
            'availability. The plant has terrestrial coverage but must survive '
            'network outages. Shared slice, Balanced cost, campus deployment.'
        ),
        scenario=(
            'You are the utility network engineer. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 120 s. Geographic scope: Campus. '
            'Isolation: Shared. Cost: Balanced. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='MC-01',
        intent=(
            'We are setting up a nationwide emergency communications network '
            'for police, fire, and ambulance. It must survive terrestrial '
            'infrastructure failures — LEO satellite fallback required. '
            'Premium, dedicated slice.'
        ),
        scenario=(
            'You are the CTO of a national public safety agency. Fallback mode: '
            'NTN_Fallback. NTN tier: LEO. Max failover: 30 seconds. Geographic '
            'scope: National. Isolation: Dedicated. Cost: Premium. Sector: '
            'Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='MC-02',
        intent=(
            'Critical infrastructure protection for a national power grid. The '
            'system must remain operational during disasters. Hybrid TN/NTN '
            'deployment across multiple regions. Balanced cost.'
        ),
        scenario=(
            'You are a grid security engineer. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 60 seconds. Geographic scope: Regional. '
            'Isolation: Dedicated. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='MC-03',
        intent=(
            'Urban command and control for city emergency services. Terrestrial '
            'network is sufficient — no satellite needed. City-wide coverage, '
            'dedicated slice, Premium.'
        ),
        scenario=(
            'You are the city IT director for emergency services. Deployment '
            'mode: TN_Only. Geographic scope: CityWide. Isolation: Dedicated. '
            'Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'TN_Only', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='MC-04',
        intent=(
            'Oil refinery process monitoring requires mission-critical '
            'connectivity with guaranteed failover to satellite. Coverage spans '
            'a large industrial campus. Latency can be up to 50 ms. Balanced '
            'cost.'
        ),
        scenario=(
            'You are the IT manager at an oil refinery. Deployment mode: '
            'Hybrid. NTN tier: LEO. Max failover: 10 seconds. Geographic scope: '
            'Campus. Isolation: Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='MC-05',
        intent=(
            'Public safety network for a disaster-prone coastal region. Must '
            'use LEO satellite when terrestrial fails. Failover must complete '
            'within 20 seconds. Regional scope, Premium.'
        ),
        scenario=(
            'You are the emergency services network planner. Deployment mode: '
            'Hybrid. NTN tier: LEO. Max failover: 20 seconds. Geographic scope: '
            'Regional. Isolation: Dedicated. Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='seed',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='EXT-RC-01',
        intent=(
            'Cargo ships on Arctic routes need broadband for navigation, '
            'weather, and crew welfare. 80 vessels, 15 Mbps minimum, latency '
            'under 500 ms. LEO satellite preferred.'
        ),
        scenario=(
            'You are the shipping company network manager. Geographic scope: '
            'Maritime. NTN tier: LEO. Min throughput: 15 Mbps. Latency budget: '
            '500 ms. Vessel count: 80. Isolation: Shared. Cost: Balanced. '
            'Sector: Transportation.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-RC-02',
        intent=(
            'Desert pipeline monitoring for a 1200 km oil pipeline crossing '
            'remote terrain. Need 2 Mbps for video surveillance and sensor '
            'data. No terrestrial coverage. LEO satellite.'
        ),
        scenario=(
            'You are the pipeline operations engineer. Geographic scope: Rural. '
            'NTN tier: LEO. Min throughput: 2 Mbps. Latency budget: 300 ms. '
            'Isolation: Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Rural', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-RC-03',
        intent=(
            'Mountain rescue coordination: teams operate above 3000 m altitude '
            'where terrestrial coverage is absent. Need reliable radio-over-IP, '
            '2 Mbps minimum, UAV-HAPS relay.'
        ),
        scenario=(
            'You are the search and rescue operations manager. Geographic '
            'scope: Aerial. NTN tier: UAV_HAPS. Min throughput: 2 Mbps. Latency '
            'budget: 200 ms. Isolation: Shared. Cost: LowCost. Sector: '
            'Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Aerial', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-RC-04',
        intent=(
            'Remote island resort with 1500 guests needs broadband internet. '
            'Island is 200 km offshore. Minimum 20 Mbps shared, latency under '
            '400 ms. Balanced cost.'
        ),
        scenario=(
            'You are the resort IT manager. Geographic scope: Maritime. NTN '
            'tier: LEO. Min throughput: 20 Mbps. Latency budget: 400 ms. '
            'Isolation: Shared. Cost: Balanced. Sector: Entertainment.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='RC-01',
        intent=(
            '150 offshore oil platforms in the North Sea need broadband '
            'connectivity. Minimum 10 Mbps per platform for crew communications '
            'and telemetry. Latency up to 300 ms is acceptable. LEO satellite '
            'preferred.'
        ),
        scenario=(
            'You are the network director for an offshore oil company. '
            'Geographic scope: Maritime. NTN tier: LEO. Min throughput: 10 '
            'Mbps. Latency budget: 300 ms. Vessel count: 150. Isolation: '
            'Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='RC-02',
        intent=(
            'Rural broadband for 200 remote villages across a mountainous '
            'region. Minimum 5 Mbps per village, latency under 500 ms is fine. '
            'LEO satellite is the only viable option.'
        ),
        scenario=(
            'You are a rural connectivity project manager. Geographic scope: '
            'Rural. NTN tier: LEO. Min throughput: 5 Mbps. Latency budget: 400 '
            'ms. Isolation: Shared. Cost: Balanced. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Rural', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='RC-03',
        intent=(
            'UAV fleet management for aerial surveying drones over national '
            'parks. Need reliable telemetry at 2 Mbps per drone, latency up to '
            '200 ms. UAV-HAPS relay preferred for on-demand coverage.'
        ),
        scenario=(
            'You are the operations manager for a drone survey company. '
            'Geographic scope: Aerial. NTN tier: UAV_HAPS. Min throughput: 2 '
            'Mbps. Latency budget: 200 ms. Isolation: Shared. Cost: LowCost. '
            'Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Aerial', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='RC-04',
        intent=(
            'Scientific research stations in Antarctica need connectivity for '
            'data uploads and video conferencing. 1 Mbps minimum, latency of '
            'several hundred ms is acceptable. GEO satellite available.'
        ),
        scenario=(
            'You are the IT manager for a polar research institute. Geographic '
            'scope: Polar. NTN tier: GEO. Min throughput: 1 Mbps. Latency '
            'budget: 600 ms. Interactive traffic: False. Isolation: Shared. '
            'Cost: Balanced. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Polar', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='RC-05',
        intent=(
            'Fishing vessels need connectivity at sea for navigation, weather '
            'updates, and crew welfare internet access. Roughly 300 vessels in '
            'the region. 10 Mbps per vessel, latency under 400 ms. LEO '
            'satellite.'
        ),
        scenario=(
            'You are the fleet manager for a fishing company. Geographic scope: '
            'Maritime. NTN tier: LEO. Min throughput: 10 Mbps. Latency budget: '
            '400 ms. Vessel count: 300. Isolation: Shared. Cost: Balanced. '
            'Sector: Maritime.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='seed',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='DC-01',
        intent=(
            'Deploying 2 million smart electricity meters across a city. Each '
            'meter reports consumption every 30 minutes. Battery-powered '
            'devices, 10-year lifetime target. 200-byte payloads.'
        ),
        scenario=(
            'You are the IoT architect for a utility company. Device density: '
            '1500/km2. Reporting interval: 1800 s. Payload: 200 bytes. Battery: '
            'yes. Target lifetime: 10 years. Geographic scope: CityWide. '
            'Isolation: Shared. Cost: LowCost. Sector: Energy.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='DC-02',
        intent=(
            'Agricultural IoT: 500,000 soil moisture and weather sensors across '
            'a national farming region. Sensors report every 15 minutes. '
            'Devices are solar+battery, small payloads of 50 bytes.'
        ),
        scenario=(
            'You are the IoT engineer for a smart farming platform. Device '
            'density: 200/km2. Reporting interval: 900 s. Payload: 50 bytes. '
            'Battery: yes. Target lifetime: 5 years. Geographic scope: '
            'Regional. Isolation: Shared. Cost: LowCost. Sector: Industry.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='DC-03',
        intent=(
            'Smart city infrastructure: 1 million parking sensors, street '
            'lights, and waste bins reporting status every hour. Very small '
            'payloads. City-wide deployment, cost must be minimal.'
        ),
        scenario=(
            'You are the smart city project manager. Device density: 5000/km2. '
            'Reporting interval: 3600 s. Payload: 20 bytes. Battery: yes. '
            'Target lifetime: 7 years. Geographic scope: CityWide. Isolation: '
            'Shared. Cost: LowCost. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='DC-04',
        intent=(
            'Industrial factory floor monitoring: 200,000 vibration sensors on '
            'machines reporting every 5 minutes. Payload is 100 bytes per '
            'report. Devices are mains-powered. Campus deployment.'
        ),
        scenario=(
            'You are the IIoT engineer at a large manufacturing plant. Device '
            'density: 10000/km2. Reporting interval: 300 s. Payload: 100 bytes. '
            'Battery: no. Geographic scope: Campus. Isolation: Shared. Cost: '
            'Balanced. Sector: Manufacturing.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='DC-05',
        intent=(
            'Water utility: smart water meters for an entire city, reporting '
            'flow every 20 minutes. 800,000 devices, battery-powered, 8-year '
            'target. Messages are 150 bytes. Shared slice is fine.'
        ),
        scenario=(
            'You are the network engineer for a water utility. Device density: '
            '1200/km2. Reporting interval: 1200 s. Payload: 150 bytes. Battery: '
            'yes. Target lifetime: 8 years. Geographic scope: CityWide. '
            'Isolation: Shared. Cost: LowCost. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-MAC-01',
        intent=(
            'Forest fire early warning: 800,000 temperature and smoke sensors '
            'across a national park, reporting every 10 minutes. Solar-powered. '
            'NTN-based coverage. LowCost.'
        ),
        scenario=(
            'You are the forestry IoT engineer. Device density: 150/km2. '
            'Reporting interval: 600 s. Payload: 30 bytes. Battery: yes. Target '
            'lifetime: 5 years. Geographic scope: Rural. Isolation: Shared. '
            'Cost: LowCost. Sector: Government.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-MAC-02',
        intent=(
            'Urban traffic management: 500,000 speed and flow sensors on roads '
            'across the city. Reporting every 30 seconds, 50-byte payloads. '
            'Shared slice, LowCost.'
        ),
        scenario=(
            'You are the city traffic engineer. Device density: 2000/km2. '
            'Reporting interval: 30 s. Payload: 50 bytes. Battery: yes. Target '
            'lifetime: 5 years. Geographic scope: CityWide. Isolation: Shared. '
            'Cost: LowCost. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-MAC-03',
        intent=(
            'Livestock tracking on a national scale: 2 million GPS tags on '
            'cattle across rural grazing land. Daily location reports. '
            'Battery-powered, minimal cost.'
        ),
        scenario=(
            'You are the agri-tech platform engineer. Device density: 100/km2. '
            'Reporting interval: 86400 s. Payload: 20 bytes. Battery: yes. '
            'Target lifetime: 3 years. Geographic scope: Regional. Isolation: '
            'Shared. Cost: LowCost. Sector: Agriculture.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-MAC-04',
        intent=(
            'River water quality monitoring network: 200,000 sensors along '
            '1,000 km of river system, reporting pH and flow every 15 minutes. '
            'Solar-powered, LowCost, NTN coverage.'
        ),
        scenario=(
            'You are the environmental monitoring engineer. Device density: '
            '200/km2. Reporting interval: 900 s. Payload: 40 bytes. Battery: '
            'yes. Target lifetime: 7 years. Geographic scope: Regional. '
            'Isolation: Shared. Cost: LowCost. Sector: Government.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='seed',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# Paraphrased variants — reworded, same requirements (12)
# ────────────────────────────────────────────────────────────────────────────

_PARAPHRASED_SAMPLES = [
    EvalSample(
        id='EXT-TC-05',
        intent=(
            'A high-precision CNC milling system for aerospace parts '
            'manufacturing must sync tool movements within 1 millisecond across '
            '10 machines. Balanced cost, dedicated 5G slice.'
        ),
        scenario=(
            'You are the manufacturing engineer at an aerospace supplier. '
            'Application: RoboticControl. Cycle time: 1 ms. Geographic scope: '
            'Campus. Cost: Balanced. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'isolation_level': 'Dedicated'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='PAR-TC-01',
        intent=(
            'Deploy a 5G slice for synchronized motion control across 60 '
            'industrial welding robots on an automobile production line. '
            'Control cycle must not exceed 1 ms. Isolated slice, Premium '
            'budget.'
        ),
        scenario=(
            'You are the OT engineer at an automotive factory. Application '
            'type: RoboticControl. Cycle time: 1 ms. Geographic scope: Campus. '
            'Cost: Premium. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl', 'isolation_level': 'Dedicated'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='PAR-TC-02',
        intent=(
            'Highway truck convoy automation needs wireless synchronization '
            'below 1 ms for safe platooning. Dedicated 5G on a regional highway '
            'corridor. Premium budget.'
        ),
        scenario=(
            'You are a transportation systems engineer. Application: V2X. V2X '
            'scenario: Platooning. Cycle time: 0.8 ms. Geographic scope: Urban. '
            'Cost: Premium. Sector: Automotive.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'V2X', 'isolation_level': 'Dedicated'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-MC-05',
        intent=(
            'National railway signalling and control network must maintain '
            'continuous operation even during regional network outages. '
            'Satellite backup required. Premium, nationwide.'
        ),
        scenario=(
            'You are the railway network engineer. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 15 s. Geographic scope: National. '
            'Isolation: Dedicated. Cost: Premium. Sector: Transportation.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='paraphrased',
        scenario_label='RuralRemote',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='PAR-MC-01',
        intent=(
            'National emergency services — police, fire, and EMS — need a '
            'dedicated resilient network. Must stay operational if ground '
            'infrastructure fails, using LEO satellite as backup. Premium cost.'
        ),
        scenario=(
            'You are the CTO of a national public safety agency. Deployment '
            'mode: Hybrid. NTN tier: LEO. Max failover: 30 seconds. Geographic '
            'scope: National. Isolation: Dedicated. Cost: Premium. Sector: '
            'Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='paraphrased',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='PAR-MC-02',
        intent=(
            'Mission-critical monitoring at a petroleum refinery must switch to '
            'satellite if the terrestrial link fails. Single-site campus, '
            'moderate cost acceptable.'
        ),
        scenario=(
            'You are the IT manager at an oil refinery. Deployment mode: '
            'Hybrid. NTN tier: LEO. Max failover: 10 seconds. Geographic scope: '
            'Campus. Isolation: Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='EXT-RC-05',
        intent=(
            'Provide internet access to 150 remote mining camps spread across a '
            '200,000 km² wilderness region. At least 10 Mbps per camp. Latency '
            'under 500 ms is acceptable. LEO satellite.'
        ),
        scenario=(
            "You are the mining company's IT manager. Geographic scope: Rural. "
            'NTN tier: LEO. Min throughput: 10 Mbps. Latency budget: 400 ms. '
            'Isolation: Shared. Cost: Balanced. Sector: Industry.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Rural', 'priority_level': 'Normal'},
        request_type='paraphrased',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='PAR-RC-01',
        intent=(
            'Over 150 oil extraction platforms in the North Sea need broadband '
            'for operations and crew. Each needs at least 10 Mbps, a few '
            'hundred milliseconds latency is fine. Prefer satellite.'
        ),
        scenario=(
            'You are the network director for an offshore oil company. '
            'Geographic scope: Maritime. NTN tier: LEO. Min throughput: 10 '
            'Mbps. Latency budget: 300 ms. Vessel count: 150. Isolation: '
            'Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='paraphrased',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='PAR-RC-02',
        intent=(
            'Bring internet to 200 isolated communities in mountainous terrain. '
            'Minimum 5 Mbps per community, latency under 500 ms. LEO satellite '
            'is the only viable option.'
        ),
        scenario=(
            'You are a rural connectivity project manager. Geographic scope: '
            'Rural. NTN tier: LEO. Min throughput: 5 Mbps. Latency budget: 400 '
            'ms. Isolation: Shared. Cost: Balanced. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Rural', 'priority_level': 'Normal'},
        request_type='paraphrased',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-MAC-05',
        intent=(
            'Rolling out 3 million smart gas meters across a national gas '
            'utility. Meters report consumption once per day, 100-byte '
            'payloads. Battery-powered, 10-year target, minimal cost.'
        ),
        scenario=(
            'You are the IoT architect for a gas utility. Device density: '
            '800/km2. Reporting interval: 86400 s. Payload: 100 bytes. Battery: '
            'yes. Target lifetime: 10 years. Geographic scope: National. '
            'Isolation: Shared. Cost: LowCost. Sector: Energy.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='PAR-DC-01',
        intent=(
            'Roll out wireless connectivity for 2 million residential '
            'electricity meters across the city. Each sends a reading every 30 '
            'minutes. Battery-powered, 10-year lifetime. Messages are 200 '
            'bytes.'
        ),
        scenario=(
            'You are the IoT architect for a utility company. Device density: '
            '1500/km2. Reporting interval: 1800 s. Payload: 200 bytes. Battery: '
            'yes. Target lifetime: 10 years. Geographic scope: CityWide. '
            'Isolation: Shared. Cost: LowCost. Sector: Energy.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='PAR-DC-02',
        intent=(
            'City-wide IoT for parking sensors, street lighting, and refuse '
            'bins — about 1 million nodes, checking in hourly. Tiny data '
            'payloads. Budget must be minimal.'
        ),
        scenario=(
            'You are the smart city project manager. Device density: 5000/km2. '
            'Reporting interval: 3600 s. Payload: 20 bytes. Battery: yes. '
            'Target lifetime: 7 years. Geographic scope: CityWide. Isolation: '
            'Shared. Cost: LowCost. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='paraphrased',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# Ambiguous variants — one or more mandatory slots omitted (12)
# ────────────────────────────────────────────────────────────────────────────

_AMBIGUOUS_SAMPLES = [
    EvalSample(
        id='AMB-TC-01',
        intent=(
            'Industrial robot control for a car manufacturing plant. Real-time '
            'closed-loop control needed. Dedicated 5G slice, Premium.'
        ),
        scenario=(
            'You are the OT engineer at an automotive factory. Application '
            'type: RoboticControl. Cycle time: 1 ms. Geographic scope: Campus. '
            'Cost: Premium. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RoboticControl'},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='AMB-TC-02',
        intent=(
            'Ultra-low latency 5G slice for critical industrial automation. 1 '
            'ms cycle time, dedicated slice, campus deployment.'
        ),
        scenario=(
            'You are a process control engineer. Application: '
            'RemoteProcessControl. Cycle time: 1 ms. Geographic scope: Campus. '
            'Cost: Balanced. Sector: Industry.'
        ),
        expected_class='TimeCritical',
        expected_slots={'cycle_time_ms': 1.0},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='EXT-TC-06',
        intent=(
            'Remote surgical assistance robot: needs ultra-low latency for '
            'precision motion control. Premium budget. Campus hospital.'
        ),
        scenario=(
            'You are the hospital CTO. Application: RemoteProcessControl. Cycle '
            'time: 1 ms. Geographic scope: Campus. Cost: Premium. Sector: '
            'Healthcare.'
        ),
        expected_class='TimeCritical',
        expected_slots={'application_type': 'RemoteProcessControl'},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='AMB-MC-01',
        intent=(
            'Highly available communications network for emergency services in '
            'a metropolitan area. Dedicated slice, Premium cost.'
        ),
        scenario=(
            'You are the city IT director for emergency services. Deployment '
            'mode: TN_Only. Geographic scope: CityWide. Isolation: Dedicated. '
            'Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'priority_level': 'High'},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='AMB-MC-02',
        intent=(
            'Mission-critical public safety network needing satellite backup '
            'for disaster resilience. Dedicated slice, Balanced cost.'
        ),
        scenario=(
            'You are the emergency services network planner. Deployment mode: '
            'Hybrid. NTN tier: LEO. Max failover: 30 seconds. Geographic scope: '
            'Regional. Isolation: Dedicated. Cost: Balanced. Sector: '
            'Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid'},
        request_type='ambiguous',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='EXT-MC-06',
        intent=(
            'Coastal patrol vessels need mission-critical communications that '
            'work offshore and near shore. Must remain connected if terrestrial '
            'fails. Dedicated, Balanced.'
        ),
        scenario=(
            'You are the coast guard network engineer. Deployment mode: Hybrid. '
            'NTN tier: LEO. Max failover: 30 s. Geographic scope: Maritime. '
            'Isolation: Dedicated. Cost: Balanced. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_slots={'deployment_mode': 'Hybrid', 'priority_level': 'High'},
        request_type='ambiguous',
        scenario_label='Maritime',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='AMB-RC-01',
        intent=(
            'Offshore fishing fleet connectivity for 200 vessels at sea. Need '
            'at least 10 Mbps per vessel, 400 ms latency is fine.'
        ),
        scenario=(
            'You are the fleet manager for a fishing company. NTN tier: LEO. '
            'Geographic scope: Maritime. Min throughput: 10 Mbps. Latency '
            'budget: 400 ms. Vessel count: 200. Isolation: Shared. Cost: '
            'Balanced. Sector: Maritime.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime'},
        request_type='ambiguous',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='AMB-RC-02',
        intent=(
            'Remote connectivity for polar research stations. GEO satellite '
            'available. Latency of several hundred ms is fine.'
        ),
        scenario=(
            'You are the IT manager for a polar research institute. NTN tier: '
            'GEO. Min throughput: 1 Mbps. Geographic scope: Polar. Latency '
            'budget: 600 ms. Interactive traffic: False. Isolation: Shared. '
            'Cost: Balanced. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Polar'},
        request_type='ambiguous',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-RC-06',
        intent=(
            'Deep-sea research submersibles need to relay sensor data via '
            'support vessel to shore. GEO satellite on the vessel, data-only '
            '(no interactive traffic), latency up to 600 ms acceptable.'
        ),
        scenario=(
            'You are the oceanographic research IT manager. NTN tier: GEO. '
            'Geographic scope: Maritime. Min throughput: 2 Mbps. Latency '
            'budget: 600 ms. Interactive traffic: False. Isolation: Shared. '
            'Cost: Balanced. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_slots={'geographic_scope': 'Maritime', 'priority_level': 'Normal'},
        request_type='ambiguous',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='AMB-DC-01',
        intent=(
            'Smart city IoT for parking and waste bin sensors. Small payloads, '
            'hourly reporting, battery-powered, 7-year target. Minimal cost.'
        ),
        scenario=(
            'You are the smart city project manager. Device density: 3000/km2. '
            'Reporting interval: 3600 s. Payload: 30 bytes. Battery: yes. '
            'Target lifetime: 7 years. Geographic scope: CityWide. Isolation: '
            'Shared. Cost: LowCost. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='AMB-DC-02',
        intent=(
            'Agricultural sensor network for 500,000 soil and weather '
            'monitoring devices across a farming region. Payloads of 50 bytes, '
            'battery-powered, 5-year lifetime.'
        ),
        scenario=(
            'You are the IoT engineer for a smart farming platform. Reporting '
            'interval: 900 s. Device density: 200/km2. Payload: 50 bytes. '
            'Battery: yes. Target lifetime: 5 years. Geographic scope: '
            'Regional. Isolation: Shared. Cost: LowCost. Sector: Agriculture.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='ambiguous',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='EXT-MAC-06',
        intent=(
            'Bridge health monitoring across a national highway network: '
            'vibration and strain sensors on 500 bridges, reporting every '
            'minute. Need minimal cost and wide coverage.'
        ),
        scenario=(
            'You are the civil infrastructure IoT engineer. Device density: '
            '1000/km2. Reporting interval: 60 s. Payload: 80 bytes. Battery: '
            'yes. Target lifetime: 8 years. Geographic scope: National. '
            'Isolation: Shared. Cost: LowCost. Sector: Government.'
        ),
        expected_class='MassiveConnectivity',
        expected_slots={'priority_level': 'Low'},
        request_type='ambiguous',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
]

# ────────────────────────────────────────────────────────────────────────────
# Contradictory variants — one embedded conflict-rule instance each (40)
# ────────────────────────────────────────────────────────────────────────────

_CONTRADICTORY_SAMPLES = [
    EvalSample(
        id='CF-01',
        intent=(
            'We need a dedicated 5G slice for a critical control system but '
            'want the absolute lowest cost possible.'
        ),
        scenario=(
            'You are an engineer who asked for the cheapest option. When the '
            'system flags a conflict between Dedicated slice and LowCost, you '
            'agree to upgrade to Balanced. Application: RoboticControl. Cycle '
            'time: 2 ms. Geographic scope: Campus. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-02',
        intent=(
            'Mission-critical emergency network with Critical SLA priority but '
            'we insist on LowCost profile to stay within tight budget.'
        ),
        scenario=(
            'You are a budget-constrained IT manager. When told '
            'Critical+LowCost is incompatible, you agree to upgrade to '
            'Balanced. Deployment mode: TN_Only. Geographic scope: Urban. '
            'Isolation: Shared. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-03',
        intent=(
            'Deploy terrestrial-only IoT connectivity for maritime buoys and '
            'coastal sea-level monitoring stations. Budget must be minimal.'
        ),
        scenario=(
            'You are the IoT engineer for a coastal monitoring authority. When '
            'the system flags TN-only is not viable in Maritime, you agree to '
            'switch to NTN_Only. NTN tier: LEO. Geographic scope: Maritime. '
            'Device density: 50/km2. Reporting interval: 3600 s. Payload: 50 '
            'bytes. Battery: yes. Isolation: Shared. Cost: LowCost. Sector: '
            'Government.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-04',
        intent=(
            'Critical robot control for an offshore platform requires a Hybrid '
            'TN/NTN deployment. The NTN layer should use GEO satellite for '
            'wide-area coverage.'
        ),
        scenario=(
            'You are the OT engineer on an offshore oil platform. When told GEO '
            'is incompatible with Critical priority, you agree to switch to '
            'LEO. Application: RoboticControl. Cycle time: 2 ms. Geographic '
            'scope: Maritime. Cost: Premium. Sector: Energy.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['GEO', 'Critical'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-05',
        intent=(
            'We need a mission-critical telemedicine network for remote surgery '
            'assistance in rural hospitals. Real-time video with sub-100 ms '
            'latency is required. GEO satellite preferred for wide national '
            'coverage.'
        ),
        scenario=(
            'You are the CTO of a telemedicine platform. When the system flags '
            'that GEO RTT (~540 ms) is incompatible with latency-sensitive '
            'surgical operations, you agree to switch to LEO. Deployment mode: '
            'Hybrid. Geographic scope: National. Max failover: 30 seconds. '
            'Isolation: Dedicated. Cost: Premium. Sector: Healthcare.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['GEO', 'latency'],
        request_type='contradictory',
        scenario_label='RuralRemote',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-06',
        intent=(
            'Offshore oil rigs in the Gulf of Mexico need live two-way video '
            'conferencing with onshore engineers and real-time interactive '
            'control. GEO satellite is already installed on all platforms.'
        ),
        scenario=(
            'You are the network manager for an offshore energy company. When '
            'told GEO RTT is incompatible with interactive traffic, you agree '
            'to switch to LEO. NTN tier: GEO. Geographic scope: Maritime. Min '
            'throughput: 10 Mbps. Latency budget: 80 ms. Interactive traffic: '
            'True. Isolation: Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['GEO', 'interactive'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-07',
        intent=(
            'Deploy 2 million smart water meters across the city with a fully '
            'dedicated Premium network slice to guarantee zero interference.'
        ),
        scenario=(
            'You are the IoT architect for a water utility. When told a '
            'dedicated Premium slice is economically unviable for massive IoT '
            'at this scale, you agree to switch to Shared isolation and LowCost '
            'profile. Device density: 1200/km2. Reporting interval: 1200 s. '
            'Payload: 150 bytes. Battery: yes. Target lifetime: 8 years. '
            'Geographic scope: CityWide. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Dedicated', 'Premium'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-08',
        intent=(
            'We need precise 1 ms robotic welding arm control for our new '
            'factory line. We are a startup — must use the lowest cost '
            'connectivity plan available.'
        ),
        scenario=(
            'You are the CTO of a robotics startup. When told Critical priority '
            'is incompatible with LowCost, you agree to upgrade to Balanced. '
            'Application: RoboticControl. Cycle time: 1 ms. Geographic scope: '
            'Campus. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-09',
        intent=(
            'V2X platooning for highway trucks: sub-millisecond communication '
            'required. Budget is very tight — we need the cheapest 5G plan '
            'available.'
        ),
        scenario=(
            'You are a fleet technology manager. When the system flags '
            'Critical+LowCost as incompatible, you accept upgrading to '
            'Balanced. Application: V2X. V2X scenario: Platooning. Cycle time: '
            '0.8 ms. Geographic scope: Urban. Sector: Automotive.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-10',
        intent=(
            'Smart grid protection relay needs trip time under 3 ms per IEC '
            '61850. The utility has a strict cost-reduction mandate — LowCost '
            'profile only.'
        ),
        scenario=(
            'You are a grid automation engineer. When told Critical latency '
            'cannot be guaranteed at LowCost, you agree to switch to Balanced. '
            'Application: GridProtection. Cycle time: 3 ms. Geographic scope: '
            'Campus. Sector: Energy.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-11',
        intent=(
            'Surgical robot remote operation requires 0.5 ms closed-loop '
            'control. The hospital procurement policy mandates using only the '
            'cheapest available connectivity tier.'
        ),
        scenario=(
            'You are the medical IT director. When the system flags that '
            'Critical SLA cannot be met at LowCost, you agree to use Balanced. '
            'Application: RemoteProcessControl. Cycle time: 0.5 ms. Geographic '
            'scope: Campus. Sector: Healthcare.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-12',
        intent=(
            'Command and control center for emergency services needs Critical '
            'SLA priority to guarantee comms during incidents. Budget freeze '
            'means we can only approve LowCost connectivity.'
        ),
        scenario=(
            'You are the IT manager for a city emergency services department. '
            'When Critical+LowCost is flagged, you accept Balanced as a '
            'compromise. Deployment mode: TN_Only. Geographic scope: CityWide. '
            'Isolation: Dedicated. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-13',
        intent=(
            'National power grid monitoring needs Critical SLA priority — '
            'outages must be detected in milliseconds. Our procurement team '
            'insists on LowCost.'
        ),
        scenario=(
            'You are a grid security engineer. When told Critical+LowCost is '
            'incompatible, you agree to Balanced. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 60 s. Geographic scope: National. '
            'Isolation: Dedicated. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Critical', 'LowCost'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-14',
        intent=(
            'Chemical plant process control needs a fully dedicated 5G slice '
            'for safety isolation. Cost must be the absolute minimum.'
        ),
        scenario=(
            'You are the process safety engineer. When told Dedicated+LowCost '
            'is not offered, you agree to switch to Balanced. Application: '
            'RemoteProcessControl. Cycle time: 2 ms. Geographic scope: Campus. '
            'Sector: Industry.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-15',
        intent=(
            'Police and fire communications require a dedicated slice to '
            'prevent interference from commercial traffic. Budget is extremely '
            'tight — we can only afford LowCost.'
        ),
        scenario=(
            'You are the public safety network manager. When told a dedicated '
            'slice requires at least Balanced, you agree to upgrade. Deployment '
            'mode: TN_Only. Geographic scope: CityWide. Max failover: 30 s. '
            'Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-16',
        intent=(
            'Offshore wind farm telemetry needs a dedicated network slice to '
            "ensure no data loss. We need the lowest cost profile — it's a "
            'renewable energy project with thin margins.'
        ),
        scenario=(
            'You are the wind farm operations engineer. When told '
            'Dedicated+LowCost is not available, you agree to Balanced. '
            'Geographic scope: Maritime. NTN tier: LEO. Isolation: Dedicated. '
            'Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-17',
        intent=(
            'We need a dedicated 5G slice for 3 million precision agriculture '
            'sensors to avoid any shared-network congestion. Budget must be '
            'LowCost.'
        ),
        scenario=(
            'You are the IoT architect for an agri-tech company. When told '
            'Dedicated+LowCost is incompatible at IoT scale, you accept Shared '
            'isolation or Balanced cost. Device density: 300/km2. Reporting '
            'interval: 900 s. Payload: 60 bytes. Battery: yes. Geographic '
            'scope: Regional. Sector: Agriculture.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-18',
        intent=(
            'Mine shaft environment monitoring: gas sensors and rock stability '
            'detectors need a dedicated slice for worker safety. Cost profile '
            'must be LowCost.'
        ),
        scenario=(
            'You are the mine safety engineer. When told a dedicated slice '
            'requires Balanced or Premium, you agree to Balanced. Application: '
            'RemoteProcessControl. Cycle time: 5 ms. Geographic scope: Campus. '
            'Sector: Industry.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['dedicated', 'LowCost'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-19',
        intent=(
            'A fleet of 300 deep-sea fishing vessels needs broadband '
            'connectivity for weather updates and navigation. They operate 400 '
            'km offshore. We want terrestrial-only 5G — no satellite.'
        ),
        scenario=(
            'You are the fleet manager. When told TN-only is not viable in '
            'Maritime scenarios, you agree to switch to NTN_Only. NTN tier: '
            'LEO. Geographic scope: Maritime. Min throughput: 10 Mbps. Latency '
            'budget: 400 ms. Isolation: Shared. Cost: Balanced. Sector: '
            'Maritime.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-20',
        intent=(
            'Ocean research vessel needs connectivity for satellite data '
            'uploads and crew welfare. The ship operates in open Pacific. '
            'Connectivity should use land towers only — no satellite.'
        ),
        scenario=(
            'You are the IT coordinator for a research institute. When told '
            'TN-only is not viable in Maritime, you agree to NTN_Only. NTN '
            'tier: LEO. Geographic scope: Maritime. Min throughput: 5 Mbps. '
            'Latency budget: 500 ms. Isolation: Shared. Cost: Balanced. Sector: '
            'Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-21',
        intent=(
            'Mission-critical communications for an offshore oil drilling '
            'platform in the North Sea. The platform is 200 km from shore. Must '
            'use terrestrial 5G only for data security policy reasons.'
        ),
        scenario=(
            'You are the platform IT manager. When told TN-only is not viable '
            '200 km offshore, you agree to Hybrid. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 15 s. Geographic scope: Maritime. '
            'Isolation: Dedicated. Cost: Premium. Sector: Energy.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-22',
        intent=(
            'Remote robotic inspection of subsea pipelines from a surface '
            'vessel. Need real-time control — vessel is in open sea. Security '
            'policy requires terrestrial-only connectivity.'
        ),
        scenario=(
            'You are the subsea operations engineer. When told TN-only is not '
            'viable for open-sea vessels, you agree to Hybrid. Application: '
            'RemoteProcessControl. Cycle time: 5 ms. Geographic scope: '
            'Maritime. Cost: Premium. Sector: Energy.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-23',
        intent=(
            'Ocean-based IoT network: 50,000 wave-energy sensors deployed '
            'across open Atlantic waters. Sensors report every hour. Data must '
            'route through terrestrial towers only.'
        ),
        scenario=(
            'You are the IoT architect for a marine energy company. When told '
            'TN-only is not viable in Maritime, you agree to NTN_Only. Device '
            'density: 20/km2. Reporting interval: 3600 s. Payload: 100 bytes. '
            'Battery: yes. Geographic scope: Maritime. Isolation: Shared. Cost: '
            'LowCost. Sector: Energy.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Maritime', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-24',
        intent=(
            'Remote CNC machining facility on an island: robotic arms need 2 ms '
            'control cycle. GEO satellite is the only connectivity option on '
            'the island.'
        ),
        scenario=(
            'You are the manufacturing engineer. When told GEO RTT (~540 ms) is '
            'incompatible with 2 ms control cycle, you agree to switch to LEO. '
            'Application: RoboticControl. Cycle time: 2 ms. Geographic scope: '
            'Maritime. Cost: Premium. Sector: Manufacturing.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['GEO', 'Critical'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-25',
        intent=(
            'Autonomous port crane at a remote Pacific island: requires 2 ms '
            'control for safe container handling. GEO satellite available for '
            'backhaul.'
        ),
        scenario=(
            'You are the port operations engineer. When told GEO satellite RTT '
            'is incompatible with Critical latency, you agree to LEO. '
            'Application: RoboticControl. Cycle time: 2 ms. Geographic scope: '
            'Campus. Cost: Balanced. Sector: Transportation.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['GEO', 'Critical'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-26',
        intent=(
            'Disaster relief command center in a remote area with no '
            'terrestrial network. Requires Critical SLA for emergency '
            'coordination. Only a GEO satellite dish is available on site.'
        ),
        scenario=(
            'You are the emergency coordinator. When told GEO RTT (~540 ms) '
            'conflicts with Critical SLA, you agree to use LEO instead. '
            'Deployment mode: NTN_Only. Geographic scope: Regional. Max '
            'failover: 30 s. Isolation: Dedicated. Cost: Premium. Sector: '
            'Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['GEO', 'Critical'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-27',
        intent=(
            'Remote patient monitoring and telemedicine for a rural hospital '
            'network. Need sub-50 ms round-trip for live diagnostics. GEO '
            'satellite available.'
        ),
        scenario=(
            'You are the telemedicine CTO. When told GEO RTT (~540 ms) is '
            'incompatible with sub-50 ms diagnostics, you agree to switch to '
            'LEO. Deployment mode: Hybrid. Geographic scope: Regional. Max '
            'failover: 60 s. Isolation: Dedicated. Cost: Premium. Sector: '
            'Healthcare.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['GEO', 'latency'],
        request_type='contradictory',
        scenario_label='RuralRemote',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-28',
        intent=(
            'Industrial automation monitoring for a remote mine: sensor data '
            'must trigger safety shutdowns within 80 ms. GEO satellite '
            'installed on site.'
        ),
        scenario=(
            'You are the mine safety IT manager. When told GEO latency '
            'conflicts with the 80 ms safety loop requirement, you agree to '
            'LEO. Deployment mode: Hybrid. Geographic scope: Regional. '
            'Isolation: Dedicated. Cost: Balanced. Sector: Industry.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['GEO', 'latency'],
        request_type='contradictory',
        scenario_label='RuralRemote',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-29',
        intent=(
            'Offshore ship-to-shore live video collaboration for 20 vessels. '
            'Engineers need real-time interactive video with onshore teams. GEO '
            'satellite already installed on all vessels.'
        ),
        scenario=(
            'You are the operations manager for an offshore engineering firm. '
            'When told GEO RTT is incompatible with interactive video, you '
            'agree to switch to LEO. NTN tier: GEO. Geographic scope: Maritime. '
            'Min throughput: 5 Mbps. Latency budget: 80 ms. Interactive '
            'traffic: True. Isolation: Shared. Cost: Balanced. Sector: Energy.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['GEO', 'interactive'],
        request_type='contradictory',
        scenario_label='Maritime',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-30',
        intent=(
            'Nationwide precision agriculture sensor network: 5 million devices '
            'monitoring soil and crop conditions. We require a fully dedicated '
            'Premium slice for guaranteed SLA.'
        ),
        scenario=(
            'You are the IoT architect for an agri-tech company. When told '
            'Dedicated+Premium is economically unviable for 5M devices, you '
            'agree to use Shared isolation and Balanced cost. Device density: '
            '500/km2. Reporting interval: 900 s. Payload: 50 bytes. Battery: '
            'yes. Target lifetime: 5 years. Geographic scope: Regional. Sector: '
            'Agriculture.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Dedicated', 'Premium'],
        request_type='contradictory',
        scenario_label='RuralRemote',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-31',
        intent=(
            'Urban smart parking: 1 million sensors reporting occupancy every 5 '
            'minutes. City council requires a dedicated Premium slice to '
            'guarantee 100% uptime for parking revenue systems.'
        ),
        scenario=(
            'You are the smart city IoT manager. When told Dedicated+Premium is '
            'not viable for million-device deployments, you agree to Shared '
            'isolation and Balanced cost. Device density: 3000/km2. Reporting '
            'interval: 300 s. Payload: 10 bytes. Battery: yes. Geographic '
            'scope: CityWide. Sector: SmartCity.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Dedicated', 'Premium'],
        request_type='contradictory',
        scenario_label='Urban',
        gold_deployment_mode='TN_Only',
    ),
    EvalSample(
        id='CF-32',
        intent=(
            'Emergency response coordination for a major earthquake zone. '
            'Mission-critical communications for first responders. We want '
            'terrestrial 5G only — no satellite dependencies.'
        ),
        scenario=(
            'You are the emergency response IT director. When told TN-only is '
            'not viable in Disaster scenarios (TN may be destroyed), you agree '
            'to use Hybrid. Deployment mode: Hybrid. NTN tier: LEO. Max '
            'failover: 30 s. Geographic scope: Regional. Isolation: Dedicated. '
            'Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-33',
        intent=(
            'Field hospital communications during a flood disaster. Medical '
            'staff need mission-critical connectivity. Policy: terrestrial 5G '
            'only for HIPAA compliance.'
        ),
        scenario=(
            'You are the field hospital IT coordinator. When told TN '
            'infrastructure may be degraded in Disaster scenarios and TN-only '
            'is not viable, you agree to Hybrid. Deployment mode: Hybrid. NTN '
            'tier: LEO. Max failover: 20 s. Geographic scope: Regional. '
            'Isolation: Dedicated. Cost: Premium. Sector: Healthcare.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-34',
        intent=(
            'Wildfire response coordination for fire crews across a large '
            'forest area. Real-time command and control needed. Terrestrial '
            'towers only please.'
        ),
        scenario=(
            'You are the wildfire operations coordinator. When told TN-only is '
            'not viable in Disaster/wildfire areas (towers may burn), you agree '
            'to Hybrid with LEO fallback. Deployment mode: Hybrid. NTN tier: '
            'LEO. Max failover: 60 s. Geographic scope: Regional. Isolation: '
            'Dedicated. Cost: Balanced. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-35',
        intent=(
            'Robotic search and rescue in a post-earthquake collapsed building '
            'area. Robot control requires 2 ms cycle time. Terrestrial '
            'infrastructure only — no satellite allowed.'
        ),
        scenario=(
            'You are the rescue robotics engineer. When told TN-only is not '
            'viable in Disaster zones (TN may be damaged), you agree to Hybrid. '
            'Application: RoboticControl. Cycle time: 2 ms. Geographic scope: '
            'Urban. Cost: Premium. Sector: Government.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-36',
        intent=(
            'Remote actuator control for emergency flood gate closure. Must '
            'operate within 1 ms for safety. City mandates terrestrial-only for '
            'all critical infrastructure.'
        ),
        scenario=(
            'You are the city infrastructure engineer. When told TN-only is not '
            'viable in Disaster scenarios, you agree to Hybrid. Application: '
            'RemoteProcessControl. Cycle time: 1 ms. Geographic scope: Urban. '
            'Cost: Balanced. Sector: Government.'
        ),
        expected_class='TimeCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-37',
        intent=(
            'Air quality and hazardous gas monitoring in a chemical plant '
            'disaster zone: 10,000 sensors need real-time reporting. '
            "Connectivity must be terrestrial only per the plant operator's "
            'security policy.'
        ),
        scenario=(
            'You are the plant safety engineer. When told TN-only is not viable '
            'in Disaster scenarios, you agree to Hybrid. Device density: '
            '5000/km2. Reporting interval: 60 s. Payload: 50 bytes. Battery: '
            'yes. Geographic scope: Campus. Isolation: Shared. Cost: LowCost. '
            'Sector: Industry.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-38',
        intent=(
            'Displaced persons camp connectivity after a major tsunami. Need '
            'broadband to bring internet to 10,000 displaced people. '
            'Terrestrial 5G towers only — satellite access not approved.'
        ),
        scenario=(
            'You are the humanitarian connectivity coordinator. When told '
            'TN-only is not viable in Disaster zones (towers may be down), you '
            'agree to NTN_Only with LEO. NTN tier: LEO. Geographic scope: '
            'Regional. Min throughput: 5 Mbps. Isolation: Shared. Cost: '
            'LowCost. Sector: Government.'
        ),
        expected_class='RemoteConnectivity',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='NTN_Only',
    ),
    EvalSample(
        id='CF-39',
        intent=(
            'Hurricane evacuation coordination for a coastal city. '
            'Mission-critical push-to-talk and video for evacuation officers. '
            'Terrestrial-only connectivity as per department policy.'
        ),
        scenario=(
            'You are the emergency management director. When told TN towers in '
            'the hurricane zone may be inoperable and TN-only is not viable, '
            'you agree to Hybrid. Deployment mode: Hybrid. NTN tier: LEO. Max '
            'failover: 30 s. Geographic scope: CityWide. Isolation: Dedicated. '
            'Cost: Premium. Sector: Government.'
        ),
        expected_class='MissionCritical',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
    EvalSample(
        id='CF-40',
        intent=(
            'Structural health monitoring for bridges in a major earthquake '
            'aftermath. 100,000 vibration sensors need to report every minute. '
            'Connectivity via terrestrial cell towers only.'
        ),
        scenario=(
            'You are the civil infrastructure IoT engineer. When told TN-only '
            'is not viable in Disaster zones, you agree to Hybrid. Device '
            'density: 2000/km2. Reporting interval: 60 s. Payload: 100 bytes. '
            'Battery: yes. Geographic scope: Regional. Isolation: Shared. Cost: '
            'LowCost. Sector: Government.'
        ),
        expected_class='MassiveConnectivity',
        expected_conflicts=['Disaster', 'TN_Only'],
        request_type='contradictory',
        scenario_label='Disaster',
        gold_deployment_mode='Hybrid',
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Collections
# ────────────────────────────────────────────────────────────────────────────

# The 12 contradictory samples included in the balanced 72-sample main set.
_SELECTED_CF_IDS = {
    'CF-01',
    'CF-02',
    'CF-03',
    'CF-04',
    'CF-05',
    'CF-06',
    'CF-07',
    'CF-09',
    'CF-16',
    'CF-19',
    'CF-30',
    'CF-32',
}

NON_CONTRADICTORY: list[EvalSample] = _SEED_SAMPLES + _PARAPHRASED_SAMPLES + _AMBIGUOUS_SAMPLES
CONTRADICTORY_SUBSET: list[EvalSample] = _CONTRADICTORY_SAMPLES
_MAIN_CF = [s for s in _CONTRADICTORY_SAMPLES if s.id in _SELECTED_CF_IDS]

ALL_SAMPLES: list[EvalSample] = NON_CONTRADICTORY + _MAIN_CF

# ── Integrity checks ──────────────────────────────────────────────────────────
assert len(NON_CONTRADICTORY) == 60, len(NON_CONTRADICTORY)
assert len(CONTRADICTORY_SUBSET) == 40, len(CONTRADICTORY_SUBSET)
assert len(_MAIN_CF) == 12, len(_MAIN_CF)
assert len(ALL_SAMPLES) == 72, len(ALL_SAMPLES)
_all_ids = [s.id for s in ALL_SAMPLES]
assert len(_all_ids) == len(set(_all_ids)), "duplicate IDs in ALL_SAMPLES"
_union_ids = {s.id for s in ALL_SAMPLES} | {s.id for s in CONTRADICTORY_SUBSET}
assert len(_union_ids) == 100, len(_union_ids)
