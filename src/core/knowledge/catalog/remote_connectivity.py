"""Group C — RemoteConnectivity: NTN-primary coverage for remote areas."""

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
