"""Paper tables encoded as data: scenarios (T1), recommendations (T4), modes (T5)."""

from models.ontology import DeploymentModeNode, ScenarioNode


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
