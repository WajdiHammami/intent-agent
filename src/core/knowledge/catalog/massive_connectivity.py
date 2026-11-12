"""Group D — MassiveConnectivity: millions of low-power IoT devices."""

from models.ontology import (
    Accompanies,
    ConflictRule,
    EscalatesTo,
    KPIProfile,
    RequiresConditional,
    RequiresParameter,
    ServiceClassNode,
    SimilarTo,
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
