"""The labeled-sample schema of the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

REQUEST_TYPES = ("seed", "paraphrased", "ambiguous", "contradictory")
WELL_FORMED_TYPES = ("seed", "paraphrased")
ADVERSARIAL_TYPES = ("ambiguous", "contradictory")

SERVICE_CLASSES = ("TimeCritical", "MissionCritical", "RemoteConnectivity", "MassiveConnectivity")
CLASS_CODES = {
    "TC": "TimeCritical", "MC": "MissionCritical",
    "RC": "RemoteConnectivity", "MAC": "MassiveConnectivity", "DC": "MassiveConnectivity",
}
CLASS_ABBREVIATIONS = {"TimeCritical": "TC", "MissionCritical": "MC",
                       "RemoteConnectivity": "RC", "MassiveConnectivity": "MaC"}


@dataclass
class EvalSample:
    """
    id                    unique sample id (e.g. CF-43)
    intent                natural-language service request
    scenario              persona script for the simulated user
    expected_class        ground-truth service class
    expected_slots        {field: value} subset that must appear in the output
    expected_conflicts    keywords expected in the detected conflict descriptions
    request_type          seed | paraphrased | ambiguous | contradictory
    scenario_label        Urban | RuralRemote | Disaster | Maritime
    gold_deployment_mode  TN_Only | NTN_Only | Hybrid ("" = not checked)
    """
    id: str
    intent: str
    scenario: str
    expected_class: str
    expected_slots: dict[str, Any] = field(default_factory=dict)
    expected_conflicts: list[str] = field(default_factory=list)
    request_type: str = "seed"
    scenario_label: str = ""
    gold_deployment_mode: str = ""

    @property
    def is_adversarial(self) -> bool:
        return self.request_type in ADVERSARIAL_TYPES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], request_type: str | None = None) -> "EvalSample":
        data = dict(data)
        if request_type is not None:
            data["request_type"] = request_type
        return cls(**data)
