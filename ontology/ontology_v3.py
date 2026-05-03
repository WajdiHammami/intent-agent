"""
TMP 3.0 — Ontology v3

4 ServiceClassNode instances with complete KG edges, plus the four deployment
scenario nodes and three deployment-mode profiles derived from the paper tables.

Paper tables encoded here:
  Table 1 — Representative TN/NTN scenarios       → SCENARIOS
  Table 2 — Service-class objectives & KPI emphasis → ServiceClassNode metadata
  Table 3 — Deployment tendencies per class         → ServiceClassNode metadata
  Table 4 — (scenario, class) → recommended mode   → DEPLOYMENT_RECOMMENDATIONS
  Table 5 — Mode performance tendencies             → DEPLOYMENT_MODES

Group taxonomy:
    A — TimeCritical        (≤1 ms latency, 99.9999% reliability)
    B — MissionCritical     (resilience-first, TN/NTN fallback)
    C — RemoteConnectivity  (NTN-primary, coverage-driven)
    D — MassiveConnectivity (≥10⁶ dev/km², ultra-low rate per device)

KPI values grounded in:
    TS 22.104, TS 22.261, TS 22.179, TS 22.280, TR 38.821,
    ITU-R M.2083, ITU-R M.2410  — see docs/KG_TMP_3.0.md §3

Universal slots (deployment_scenario, geographic_scope, isolation_level,
cost_profile) are injected by KGOntology as base requirements for every class.
vertical_domain is treated as extractable and NOT in requires.
"""

from ontology.types import (
    Accompanies,
    ConflictRule,
    DeploymentModeNode,
    EscalatesTo,
    KPIProfile,
    RequiresConditional,
    RequiresParameter,
    ScenarioNode,
    ServiceClassNode,
    SimilarTo,
)


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Deployment Scenarios
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, ScenarioNode] = {
    "Urban": ScenarioNode(
        id="Urban",
        tn_availability="High",
        ntn_availability="Available",
        characteristics="Dense users, strong TN, latency-sensitive services",
    ),
    "RuralRemote": ScenarioNode(
        id="RuralRemote",
        tn_availability="Low-Medium",
        ntn_availability="High",
        characteristics="Sparse TN, coverage extension required",
    ),
    "Disaster": ScenarioNode(
        id="Disaster",
        tn_availability="Degraded",
        ntn_availability="High",
        characteristics="Unstable TN, resilience and fallback required",
    ),
    "Maritime": ScenarioNode(
        id="Maritime",
        tn_availability="Very Low / None",
        ntn_availability="High",
        characteristics="NTN-dominant support, wide-area reach required",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Table 5 — Deployment Mode Profiles
# ══════════════════════════════════════════════════════════════════════════════

DEPLOYMENT_MODES: dict[str, DeploymentModeNode] = {
    "TN_Only": DeploymentModeNode(
        id="TN_Only",
        coverage="High in Urban, weak in sparse regions",
        availability="High when TN exists",
        complexity_cost="Low-Medium",
    ),
    "NTN_Only": DeploymentModeNode(
        id="NTN_Only",
        coverage="Broad / wide-area",
        availability="High in remote areas",
        complexity_cost="Medium-High",
    ),
    "Hybrid": DeploymentModeNode(
        id="Hybrid",
        coverage="Strongest overall",
        availability="Highest overall",
        complexity_cost="High",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — (scenario, service_class) → (recommended_mode, reason)
# ══════════════════════════════════════════════════════════════════════════════

DEPLOYMENT_RECOMMENDATIONS: dict[tuple[str, str], tuple[str, str]] = {
    # Urban
    ("Urban", "TimeCritical"):
        ("TN_Only / Hybrid",
         "Strong TN supports low latency; Hybrid adds resilience if required"),
    ("Urban", "MissionCritical"):
        ("Hybrid",
         "Mission continuity benefits from fallback support"),
    ("Urban", "RemoteConnectivity"):
        ("TN_Only / Hybrid",
         "TN is usually sufficient, Hybrid improves continuity"),
    ("Urban", "MassiveConnectivity"):
        ("TN_Only / Hybrid",
         "Dense terrestrial support may suffice, Hybrid helps load balancing"),
    # Rural / Remote
    ("RuralRemote", "TimeCritical"):
        ("Hybrid",
         "TN may be insufficient for stable support"),
    ("RuralRemote", "MissionCritical"):
        ("Hybrid",
         "Reliability and fallback are important under weak TN"),
    ("RuralRemote", "RemoteConnectivity"):
        ("Hybrid / NTN_Only",
         "Coverage extension is the main objective"),
    ("RuralRemote", "MassiveConnectivity"):
        ("NTN_Only / Hybrid",
         "Wide-area scalable connectivity favors NTN support"),
    # Disaster
    ("Disaster", "TimeCritical"):
        ("Hybrid",
         "Critical support with degraded TN requires fallback"),
    ("Disaster", "MissionCritical"):
        ("Hybrid",
         "Mission continuity and resilience dominate"),
    ("Disaster", "RemoteConnectivity"):
        ("Hybrid / NTN_Only",
         "Coverage and continuity under disruption favor NTN involvement"),
    ("Disaster", "MassiveConnectivity"):
        ("Hybrid / NTN_Only",
         "NTN can maintain service reach when TN is unstable"),
    # Maritime
    ("Maritime", "TimeCritical"):
        ("Hybrid / NTN_Only",
         "Offshore TN is limited; Hybrid only near coast"),
    ("Maritime", "MissionCritical"):
        ("NTN_Only / Hybrid",
         "Fallback and continuity rely primarily on NTN"),
    ("Maritime", "RemoteConnectivity"):
        ("NTN_Only",
         "Wide-area remote support is NTN-dominant"),
    ("Maritime", "MassiveConnectivity"):
        ("NTN_Only",
         "Massive telemetry and sensing require broad NTN coverage"),
}


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A — TimeCritical
# ══════════════════════════════════════════════════════════════════════════════

TIME_CRITICAL = ServiceClassNode(
    id="TimeCritical",
    group="A",
    display_name="Time-Critical Control",
    description=(
        "Deterministic, sub-millisecond control for industrial robots, "
        "grid protection relays, and vehicular coordination. "
        "Latency and reliability are non-negotiable. Always dedicated slice."
    ),
    typical_verticals=["Manufacturing", "Industry", "Energy", "Automotive"],
    kpi_profile=KPIProfile(
        latency_max_ms=1.0,         # TS 22.104 §4.2: closed-loop control 1 ms
        reliability=0.999999,       # TS 22.104: 10^-6 error rate
        availability=0.99999,       # TS 22.261 Table 7.2-1: five nines
        throughput_min_mbps=10.0,
        connection_density_per_km2=1_000,
    ),
    traffic_pattern="DeterministicPeriodic",
    implied_priority="Critical",
    implied_isolation="Dedicated",

    # Table 2
    primary_objective="Stable low-latency service support",
    dominant_kpi=["Latency", "Reliability"],
    # Table 3
    deployment_tendency=(
        "TN-only in strong urban TN; Hybrid when resilience is needed"
    ),

    requires=[
        RequiresParameter(
            field="deployment_mode",
            dtype="enum",
            enum_values=["TN_Only", "Hybrid"],
            mandatory=False,
            question=(
                "What deployment mode is preferred? "
                "TN-only (standard — strong TN, lowest latency) or "
                "Hybrid (adds NTN resilience where TN may be insufficient)? "
                "Note: NTN-only is not supported for time-critical control."
            ),
        ),
        RequiresParameter(
            field="application_type",
            dtype="enum",
            enum_values=["RoboticControl", "GridProtection", "V2X", "RemoteProcessControl"],
            question=(
                "What type of time-critical application is this? "
                "(RoboticControl, GridProtection, V2X, or RemoteProcessControl)"
            ),
        ),
        RequiresParameter(
            field="cycle_time_ms",
            dtype="float",
            unit="ms",
            question="What is the required control loop cycle time in milliseconds?",
        ),
    ],

    requires_conditional=[
        # NTN-assisted deployment: which NTN tier provides the resilience layer?
        RequiresConditional(
            field="ntn_tier",
            dtype="enum",
            condition_field="deployment_scenario",
            condition_op="in",
            condition_value=["RuralRemote", "Disaster", "Maritime"],
            enum_values=["LEO", "UAV_HAPS"],
            question=(
                "Which NTN tier provides the resilience layer in this NTN-assisted deployment? "
                "(LEO for broad coverage, UAV/HAPS for on-demand local coverage) "
                "GEO is not supported for time-critical control due to latency."
            ),
        ),
        # V2X: scenario drives deployment sizing
        RequiresConditional(
            field="v2x_scenario",
            dtype="enum",
            condition_field="application_type",
            condition_op="eq",
            condition_value="V2X",
            enum_values=["Intersection", "Highway", "Platooning"],
            question="What V2X scenario applies? (Intersection, Highway, or Platooning)",
        ),
        # Grid protection: trip time drives IEC 61850 protection class selection
        RequiresConditional(
            field="protection_trip_time_ms",
            dtype="float",
            condition_field="application_type",
            condition_op="eq",
            condition_value="GridProtection",
            unit="ms",
            question=(
                "What is the required protection relay trip time in ms? "
                "(IEC 61850 class P1 defines ≤4 ms)"
            ),
        ),
        # Sub-ms cycle: clock sync accuracy becomes mandatory
        RequiresConditional(
            field="sync_accuracy_us",
            dtype="float",
            condition_field="cycle_time_ms",
            condition_op="lt",
            condition_value=1.0,
            unit="µs",
            question="What clock synchronization accuracy is required in microseconds?",
        ),
    ],

    conflicts=[
        # NTN-only is physically impossible for ≤1 ms latency
        ConflictRule(
            subject_field="ntn_tier",
            subject_value="GEO",
            object_field="priority_level",
            object_value="Critical",
            message=(
                "GEO satellite RTT (~540 ms) cannot meet Critical-priority "
                "latency (≤1 ms). Use LEO or UAV/HAPS for the NTN resilience layer."
            ),
        ),
    ],

    escalates_to=[
        # TimeCritical is Group A — top group, no escalation out
    ],

    similar_to=[
        SimilarTo(
            target_class="MissionCritical",
            trigger_field="cycle_time_ms",
            trigger_value=5.0,
            similarity_score=0.8,
        ),
    ],

    accompanies=[
        Accompanies(
            companion_class="MassiveConnectivity",
            scenario=(
                "Smart factory: TimeCritical handles robotic arm control while "
                "MassiveConnectivity covers the environmental sensor mesh."
            ),
        ),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — MissionCritical
# ══════════════════════════════════════════════════════════════════════════════

MISSION_CRITICAL = ServiceClassNode(
    id="MissionCritical",
    group="B",
    display_name="Mission-Critical Connectivity",
    description=(
        "Resilience-first services where continuity and availability matter "
        "more than sub-millisecond latency. TN/NTN deployment mode is central. "
        "Public safety, emergency response, critical infrastructure protection."
    ),
    typical_verticals=["Government", "Energy", "Transportation", "Industry"],
    kpi_profile=KPIProfile(
        latency_max_ms=10.0,        # TS 22.261 §6.2: discrete automation 10 ms
        reliability=0.9999,         # TS 22.179 §7.2: MCPTT four nines
        availability=0.99999,       # TS 22.280 §5: public safety five nines
        throughput_min_mbps=2.0,
        connection_density_per_km2=500,
    ),
    traffic_pattern="EventDriven",
    implied_priority="High",

    # Table 2
    primary_objective="Resilient and continuous service support",
    dominant_kpi=["Reliability", "Fallback", "Availability"],
    # Table 3
    deployment_tendency=(
        "Hybrid in most cases; NTN fallback under degraded TN"
    ),

    requires=[
        RequiresParameter(
            field="deployment_mode",
            dtype="enum",
            enum_values=["TN_Only", "NTN_Only", "Hybrid"],
            mandatory=False,
            question=(
                "What deployment mode is preferred for this mission-critical service? "
                "TN-only (reliable terrestrial exists), "
                "NTN-only (remote/maritime, no terrestrial), or "
                "Hybrid (maximum resilience — recommended for most scenarios)?"
            ),
        ),
        # isolation_level and cost_profile injected universally
    ],

    requires_conditional=[
        # NTN involved: which tier?
        RequiresConditional(
            field="ntn_tier",
            dtype="enum",
            condition_field="deployment_scenario",
            condition_op="in",
            condition_value=["RuralRemote", "Disaster", "Maritime"],
            enum_values=["LEO", "GEO", "UAV_HAPS", "Any"],
            question=(
                "Which NTN tier should be used? "
                "(LEO for low latency, GEO for wide area, UAV_HAPS for on-demand, Any)"
            ),
        ),
        # NTN involved: switchover time tolerance
        RequiresConditional(
            field="max_failover_time_s",
            dtype="float",
            condition_field="deployment_scenario",
            condition_op="in",
            condition_value=["RuralRemote", "Disaster", "Maritime"],
            unit="s",
            question="What is the maximum acceptable failover time from TN to NTN in seconds?",
        ),
        # GEO selected: probe latency sensitivity
        RequiresConditional(
            field="latency_sensitive",
            dtype="bool",
            condition_field="ntn_tier",
            condition_op="eq",
            condition_value="GEO",
            question=(
                "GEO round-trip latency is ~540 ms. Is sub-300 ms latency required? "
                "(yes/no — if yes, GEO is incompatible)"
            ),
        ),
    ],

    conflicts=[
        # GEO + latency-sensitive: physically impossible
        ConflictRule(
            subject_field="ntn_tier",
            subject_value="GEO",
            object_field="latency_sensitive",
            object_value="True",
            message=(
                "GEO satellite RTT is ~540 ms — incompatible with "
                "latency-sensitive mission-critical operations. Use LEO or UAV/HAPS."
            ),
        ),
        # Shared slice + TN-only: cannot guarantee five-nines availability
        ConflictRule(
            subject_field="isolation_level",
            subject_value="Shared",
            object_field="deployment_mode",
            object_value="TN_Only",
            message=(
                "TN-only shared slice cannot guarantee 99.999% availability for "
                "mission-critical services. Use a Dedicated slice or add NTN fallback."
            ),
        ),
    ],

    escalates_to=[
        EscalatesTo(
            target_class="TimeCritical",
            condition_field="cycle_time_ms",
            condition_op="lt",
            condition_value=5.0,
            reason=(
                "Control loop < 5 ms exceeds MissionCritical scope (≤10 ms). "
                "TimeCritical (Group A) is required."
            ),
        ),
    ],

    similar_to=[
        SimilarTo(
            target_class="TimeCritical",
            trigger_field="isolation_level",
            trigger_value="Dedicated",
            similarity_score=0.7,
        ),
        SimilarTo(
            target_class="RemoteConnectivity",
            trigger_field="deployment_mode",
            trigger_value="NTN_Only",
            similarity_score=0.65,
        ),
    ],

    accompanies=[
        Accompanies(
            companion_class="RemoteConnectivity",
            scenario=(
                "Disaster response: MissionCritical handles local command/control "
                "while RemoteConnectivity provides the NTN backhaul to the operations center."
            ),
        ),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — RemoteConnectivity
# ══════════════════════════════════════════════════════════════════════════════

REMOTE_CONNECTIVITY = ServiceClassNode(
    id="RemoteConnectivity",
    group="C",
    display_name="Remote / NTN Connectivity",
    description=(
        "NTN-primary connectivity for users and infrastructure in areas "
        "unreachable by terrestrial networks. Coverage is the primary driver. "
        "Maritime, rural, aerial, polar, and city-fringe deployments."
    ),
    typical_verticals=["Maritime", "Energy", "Government", "SmartCity", "Transportation"],
    kpi_profile=KPIProfile(
        latency_max_ms=500.0,       # TR 38.821 §5.2: LEO RTT budget
        reliability=0.999,
        availability=0.999,
        throughput_min_mbps=1.0,
        connection_density_per_km2=10,
    ),
    traffic_pattern="Continuous",
    implied_priority="Normal",

    # Table 2
    primary_objective="Wide-area service reach",
    dominant_kpi=["Coverage", "Continuity"],
    # Table 3
    deployment_tendency=(
        "Hybrid or NTN-only, especially in sparse / remote scenarios"
    ),

    requires=[
        RequiresParameter(
            field="deployment_mode",
            dtype="enum",
            enum_values=["NTN_Only", "Hybrid"],
            mandatory=False,
            question=(
                "What deployment mode is preferred? "
                "NTN-only (pure satellite/HAPS, no terrestrial), or "
                "Hybrid (NTN augmented by terrestrial near populated areas)? "
                "Note: TN-only is not applicable for remote connectivity."
            ),
        ),
        # geographic_scope for RC uses its own enum (overrides universal)
        RequiresParameter(
            field="geographic_scope",
            dtype="enum",
            enum_values=["Maritime", "Rural", "Aerial", "Polar", "CityFringe"],
            question=(
                "What type of remote environment is this? "
                "(Maritime, Rural, Aerial/UAV operations, Polar, or CityFringe)"
            ),
        ),
        RequiresParameter(
            field="ntn_tier",
            dtype="enum",
            enum_values=["LEO", "GEO", "UAV_HAPS", "Any"],
            question=(
                "What NTN tier is preferred? "
                "(LEO for low latency, GEO for wide area, UAV_HAPS for on-demand, Any)"
            ),
        ),
        RequiresParameter(
            field="min_throughput_mbps",
            dtype="float",
            unit="Mbps",
            question="What minimum throughput per terminal is required in Mbps?",
        ),
        RequiresParameter(
            field="latency_budget_ms",
            dtype="float",
            unit="ms",
            question="What is the maximum acceptable end-to-end latency in milliseconds?",
        ),
    ],

    requires_conditional=[
        # Maritime: vessel count drives capacity planning
        RequiresConditional(
            field="vessel_count",
            dtype="int",
            condition_field="geographic_scope",
            condition_op="eq",
            condition_value="Maritime",
            question="Approximately how many vessels or offshore platforms are in the deployment area?",
        ),
        # GEO: confirm whether interactive traffic is expected
        RequiresConditional(
            field="interactive_traffic",
            dtype="bool",
            condition_field="ntn_tier",
            condition_op="eq",
            condition_value="GEO",
            question=(
                "Will this link carry interactive or real-time traffic? "
                "(GEO latency ~540 ms may be prohibitive for interactive use)"
            ),
        ),
        # High throughput: terminal type matters for feasibility
        RequiresConditional(
            field="terminal_type",
            dtype="enum",
            condition_field="min_throughput_mbps",
            condition_op="gt",
            condition_value=50.0,
            enum_values=["VSAT", "LEO_FlatPanel", "NR_NTN_UE"],
            question=(
                "What terminal type supports this throughput requirement? "
                "(VSAT, LEO flat-panel antenna, or integrated NR-NTN UE)"
            ),
        ),
    ],

    conflicts=[
        ConflictRule(
            subject_field="ntn_tier",
            subject_value="GEO",
            object_field="interactive_traffic",
            object_value="True",
            message=(
                "GEO RTT is ~540 ms — incompatible with interactive or real-time "
                "traffic. Use LEO (RTT 10-40 ms) or UAV/HAPS instead."
            ),
        ),
        ConflictRule(
            subject_field="isolation_level",
            subject_value="Dedicated",
            object_field="cost_profile",
            object_value="LowCost",
            message=(
                "Dedicated NTN slice with LowCost profile is not offered. "
                "NTN capacity costs require at least a Balanced profile."
            ),
        ),
    ],

    escalates_to=[
        EscalatesTo(
            target_class="MissionCritical",
            condition_field="interactive_traffic",
            condition_op="eq",
            condition_value=True,
            reason=(
                "Interactive real-time traffic over NTN with resilience requirements "
                "describes MissionCritical (Group B), not RemoteConnectivity."
            ),
        ),
    ],

    similar_to=[
        SimilarTo(
            target_class="MassiveConnectivity",
            trigger_field="min_throughput_mbps",
            trigger_value=0.1,
            similarity_score=0.7,
        ),
        SimilarTo(
            target_class="MissionCritical",
            trigger_field="latency_budget_ms",
            trigger_value=50.0,
            similarity_score=0.65,
        ),
    ],

    accompanies=[
        Accompanies(
            companion_class="MassiveConnectivity",
            scenario=(
                "Remote IoT backhaul: RemoteConnectivity provides the NTN uplink "
                "while MassiveConnectivity manages the dense local sensor mesh."
            ),
        ),
        Accompanies(
            companion_class="MissionCritical",
            scenario=(
                "Disaster response: RemoteConnectivity is the NTN backhaul layer "
                "for a MissionCritical command-and-control deployment."
            ),
        ),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D — MassiveConnectivity
# ══════════════════════════════════════════════════════════════════════════════

MASSIVE_CONNECTIVITY = ServiceClassNode(
    id="MassiveConnectivity",
    group="D",
    display_name="Massive IoT Connectivity",
    description=(
        "Networks of enormous numbers of low-power, low-rate devices. "
        "Device density and energy efficiency dominate; latency tolerance is high. "
        "Smart cities, industrial sensor meshes, precision agriculture, smart metering."
    ),
    typical_verticals=["SmartCity", "Manufacturing", "Industry", "Energy", "Agriculture"],
    kpi_profile=KPIProfile(
        latency_max_ms=10_000.0,
        reliability=0.999,
        availability=0.999,
        throughput_min_mbps=0.001,
        connection_density_per_km2=1_000_000,   # ITU-R M.2083 §4.4
    ),
    traffic_pattern="Sporadic",
    implied_priority="Low",

    # Table 2
    primary_objective="Scalable sensing and telemetry support",
    dominant_kpi=["Coverage", "Connectivity Scale", "Efficiency"],
    # Table 3
    deployment_tendency=(
        "NTN or Hybrid depending on coverage and density needs"
    ),

    requires=[
        RequiresParameter(
            field="deployment_mode",
            dtype="enum",
            enum_values=["TN_Only", "NTN_Only", "Hybrid"],
            mandatory=False,
            question=(
                "What deployment mode is preferred? "
                "TN-only (dense urban, terrestrial sufficient), "
                "NTN-only (wide-area / rural, NTN for broad reach), or "
                "Hybrid (best coverage + load balancing)?"
            ),
        ),
        RequiresParameter(
            field="device_density_per_km2",
            dtype="int",
            unit="dev/km²",
            question="What is the estimated device density in devices per km²?",
        ),
        RequiresParameter(
            field="reporting_interval_s",
            dtype="float",
            unit="s",
            question="How often does each device report data, in seconds?",
        ),
        RequiresParameter(
            field="payload_size_bytes",
            dtype="int",
            unit="bytes",
            question="What is the typical payload size per message in bytes?",
        ),
    ],

    requires_conditional=[
        # NTN involved: which tier fits the coverage/latency need?
        RequiresConditional(
            field="ntn_tier",
            dtype="enum",
            condition_field="deployment_scenario",
            condition_op="in",
            condition_value=["RuralRemote", "Disaster", "Maritime"],
            enum_values=["LEO", "GEO", "UAV_HAPS", "Any"],
            question=(
                "Which NTN tier is preferred for wide-area IoT coverage? "
                "(LEO for global reach, GEO for wide-area, UAV/HAPS for on-demand, Any)"
            ),
        ),
        # Extreme density: radio technology choice becomes mandatory
        RequiresConditional(
            field="radio_technology",
            dtype="enum",
            condition_field="device_density_per_km2",
            condition_op="gt",
            condition_value=100_000,
            enum_values=["NB_IoT", "eMTC", "NR_RedCap"],
            question=(
                "At >100k dev/km², NB-IoT, eMTC, or NR RedCap is typically required. "
                "Which radio technology is preferred?"
            ),
        ),
        # Sub-minute reporting: battery life question
        RequiresConditional(
            field="battery_powered",
            dtype="bool",
            condition_field="reporting_interval_s",
            condition_op="lt",
            condition_value=60.0,
            question=(
                "Are the devices battery-powered? "
                "Sub-minute reporting significantly reduces battery lifetime."
            ),
        ),
        # Battery-powered: lifetime target
        RequiresConditional(
            field="target_battery_years",
            dtype="float",
            condition_field="battery_powered",
            condition_op="eq",
            condition_value=True,
            unit="years",
            question=(
                "What is the target battery lifetime in years? "
                "(NB-IoT supports up to 10 years at low duty cycle per TS 22.261)"
            ),
        ),
    ],

    conflicts=[
        ConflictRule(
            subject_field="isolation_level",
            subject_value="Dedicated",
            object_field="cost_profile",
            object_value="Premium",
            message=(
                "Premium dedicated slice for massive IoT is economically "
                "unviable at scale — use Shared or LowCost/Balanced."
            ),
        ),
        ConflictRule(
            subject_field="priority_level",
            subject_value="Low",
            object_field="cost_profile",
            object_value="Premium",
            message=(
                "Premium cost profile is inconsistent with Low-priority IoT. "
                "Use LowCost or Balanced."
            ),
        ),
    ],

    escalates_to=[
        EscalatesTo(
            target_class="MissionCritical",
            condition_field="reporting_interval_s",
            condition_op="lt",
            condition_value=1.0,
            reason=(
                "Sub-second reporting with reliability requirements suggests "
                "mission-critical telemetry — consider MissionCritical (Group B)."
            ),
        ),
        EscalatesTo(
            target_class="RemoteConnectivity",
            condition_field="device_density_per_km2",
            condition_op="lt",
            condition_value=100,
            reason=(
                "Density below 100 dev/km² over a large area suggests a coverage "
                "gap rather than a massive connectivity challenge. "
                "Consider RemoteConnectivity (Group C)."
            ),
        ),
    ],

    similar_to=[
        SimilarTo(
            target_class="RemoteConnectivity",
            trigger_field="geographic_scope",
            trigger_value=["Regional", "National"],
            similarity_score=0.6,
        ),
    ],

    accompanies=[
        Accompanies(
            companion_class="TimeCritical",
            scenario=(
                "Smart factory: TimeCritical controls the robotic arms while "
                "MassiveConnectivity handles the dense environmental sensor mesh."
            ),
        ),
        Accompanies(
            companion_class="RemoteConnectivity",
            scenario=(
                "Remote IoT: MassiveConnectivity manages the local sensor mesh "
                "while RemoteConnectivity provides the NTN backhaul."
            ),
        ),
    ],
)


# ── All classes — imported by KGOntology ──────────────────────────────────────

ALL_CLASSES: list[ServiceClassNode] = [
    TIME_CRITICAL,
    MISSION_CRITICAL,
    REMOTE_CONNECTIVITY,
    MASSIVE_CONNECTIVITY,
]
