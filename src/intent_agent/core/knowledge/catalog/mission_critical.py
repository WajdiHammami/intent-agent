"""Group B — MissionCritical: resilience-first connectivity with TN/NTN fallback."""

from intent_agent.models.ontology import (
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
