"""Group A — TimeCritical: deterministic sub-millisecond control."""

from intent_agent.models.ontology import (
    Accompanies,
    ConflictRule,
    KPIProfile,
    RequiresConditional,
    RequiresParameter,
    ServiceClassNode,
    SimilarTo,
)


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
