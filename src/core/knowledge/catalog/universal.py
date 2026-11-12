"""Slots and conflict rules that apply to every service class."""

from models.ontology import ConflictRule, RequiresParameter


# Synthetic slot for vertical_domain — almost always extractable from intent,
# but needed as a fallback question when Stage 2 extraction fails.
VERTICAL_DOMAIN_PARAM = RequiresParameter(
    field="vertical_domain",
    dtype="enum",
    enum_values=[
        "Manufacturing", "Industry", "Energy", "Transportation",
        "Government", "Media", "Entertainment", "SmartCity", "Automotive",
        "Maritime", "Agriculture",
    ],
    question=(
        "What vertical industry or sector does this deployment serve? "
        "(Manufacturing, Industry, Energy, Transportation, Government, "
        "Media, Entertainment, SmartCity, Automotive, Maritime, or Agriculture)"
    ),
)

# Universal slots applied to every class unless overridden by the class node.
UNIVERSAL_REQUIRES: list[RequiresParameter] = [
    RequiresParameter(
        field="deployment_scenario",
        dtype="enum",
        enum_values=["Urban", "RuralRemote", "Disaster", "Maritime"],
        question=(
            "What is the deployment scenario context? "
            "Urban (strong TN), RuralRemote (sparse TN, coverage extension needed), "
            "Disaster (degraded TN, resilience critical), or "
            "Maritime (very low/no TN, NTN-dominant)?"
        ),
    ),
    RequiresParameter(
        field="geographic_scope",
        dtype="enum",
        enum_values=["Campus", "Urban", "CityWide", "Regional", "National",
                     "Maritime", "Rural", "Aerial", "Polar", "CityFringe"],
        question=(
            "What is the geographic coverage needed? "
            "(Campus, Urban, CityWide, Regional, National, "
            "Maritime, Rural, Aerial, Polar, or CityFringe)"
        ),
    ),
    RequiresParameter(
        field="isolation_level",
        dtype="enum",
        enum_values=["Shared", "Dedicated"],
        question=(
            "Do you need a dedicated network slice (higher cost, guaranteed SLA) "
            "or is a shared slice acceptable?"
        ),
    ),
    RequiresParameter(
        field="cost_profile",
        dtype="enum",
        enum_values=["LowCost", "Balanced", "Premium"],
        question=(
            "What cost profile fits your deployment? "
            "(LowCost / Balanced / Premium)"
        ),
    ),
]

# Universal conflict rules checked for every class.
UNIVERSAL_CONFLICTS: list[ConflictRule] = [
    ConflictRule(
        subject_field="priority_level",
        subject_value="Critical",
        object_field="cost_profile",
        object_value="LowCost",
        message=(
            "Critical priority is incompatible with a LowCost profile — "
            "the SLA cannot be guaranteed. Use Balanced or Premium."
        ),
    ),
    ConflictRule(
        subject_field="isolation_level",
        subject_value="Dedicated",
        object_field="cost_profile",
        object_value="LowCost",
        message=(
            "A dedicated slice with LowCost profile is not offered — "
            "dedicated slices require at least a Balanced cost profile."
        ),
    ),
    # Maritime scenario + TN-only mode: TN availability is very low / none
    ConflictRule(
        subject_field="deployment_scenario",
        subject_value="Maritime",
        object_field="deployment_mode",
        object_value="TN_Only",
        message=(
            "TN-only deployment is not viable in Maritime scenarios where "
            "terrestrial network availability is very low or absent. "
            "Use NTN-only or Hybrid."
        ),
    ),
    # GEO tier + Critical priority: ~540 ms RTT cannot meet ≤1 ms / ≤10 ms
    ConflictRule(
        subject_field="ntn_tier",
        subject_value="GEO",
        object_field="priority_level",
        object_value="Critical",
        message=(
            "GEO satellite RTT (~540 ms) cannot meet Critical-priority latency "
            "requirements. Use LEO (10-40 ms RTT) or UAV/HAPS."
        ),
    ),
    # Disaster scenario + TN-only mode: TN is degraded/unreliable
    ConflictRule(
        subject_field="deployment_scenario",
        subject_value="Disaster",
        object_field="deployment_mode",
        object_value="TN_Only",
        message=(
            "TN-only deployment is not viable in Disaster scenarios where "
            "terrestrial infrastructure may be degraded or unavailable. "
            "Use Hybrid or NTN-only to ensure resilience."
        ),
    ),
]
