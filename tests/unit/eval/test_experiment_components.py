"""Building blocks of the experiments, without running a model."""

from adapters.llm.scripted import ScriptedLanguageModel
from adapters.users.scripted import ScriptedUser
from core.knowledge import KGConflictDetector, NullConflictDetector
from core.pipeline import IntentPipeline
from eval.dataset import get_sample
from eval.experiments.ablation.llm_conflicts import LLMConflictDetector
from eval.experiments.ablation.variants import VARIANTS
from eval.experiments.generate.builder import build_sample, next_id
from eval.experiments.pure_llm.nodes import NODE_OVERRIDES
from models.dialogue import SlotValue


class JsonModel(ScriptedLanguageModel):
    """Scripted model whose chat_json answers come from a function of the prompt."""
    def __init__(self, reply, **kw):
        super().__init__(**kw)
        self.reply = reply

    def chat_json(self, system, user, max_tokens=256, temperature=0.0):
        return self.reply(system, user)


def test_ablation_variants_configure_the_pipeline(kg):
    llm = ScriptedLanguageModel()
    assert isinstance(VARIANTS["baseline"].pipeline_options(llm, kg)["conflict_detector"], KGConflictDetector)
    assert isinstance(VARIANTS["no_kg"].pipeline_options(llm, kg)["conflict_detector"], NullConflictDetector)
    assert VARIANTS["no_s2"].pipeline_options(llm, kg)["extraction"] is False
    assert isinstance(VARIANTS["llm_conflicts"].pipeline_options(llm, kg)["conflict_detector"], LLMConflictDetector)


def test_llm_conflict_detector_parses_flags():
    llm = JsonModel(lambda s, u: {"has_conflict": True, "conflicts": [
        {"field_a": "priority_level", "field_b": "cost_profile", "description": "too cheap"}]})
    flags = LLMConflictDetector(llm).detect("TimeCritical", {"a": SlotValue("a", 1, "extracted", "high")})
    assert [(f.description, f.object_field) for f in flags] == [("too cheap", "cost_profile")]


def test_llm_conflict_detector_is_silent_on_errors_and_empty_input():
    def boom(s, u):
        raise ValueError("bad json")
    assert LLMConflictDetector(JsonModel(boom)).detect("TimeCritical", {"a": SlotValue("a", 1, "x", "high")}) == []
    assert LLMConflictDetector(JsonModel(boom)).detect("TimeCritical", {}) == []


def test_pure_llm_pipeline_lets_the_llm_decide():
    replies = iter([
        {"is_complete": False, "missing_fields": ["cost_profile"], "questions": {"cost_profile": "Budget?"},
         "conflicts": []},
        {"value": "Premium"},                                   # parse the answer
        {"is_complete": True, "missing_fields": [], "conflicts": []},
        {"deployment_mode": "Teleport"},                        # invalid → Hybrid
    ])
    llm = JsonModel(lambda s, u: next(replies), class_id="MissionCritical")
    user = ScriptedUser(["Premium"])
    result = IntentPipeline(llm, user, node_overrides=NODE_OVERRIDES).run("x")
    assert [q.text for q in user.questions] == ["Budget?"]
    assert result.state.satisfied["cost_profile"].value == "Premium"
    assert result.service_intent.class_slots["deployment_mode"] == "Hybrid"


def test_generated_ids_avoid_collisions():
    assert next_id("TC-01", "paraphrased", 1, set()) == "TC-PAR-01"
    assert next_id("TC-01", "paraphrased", 1, {"TC-PAR-01"}) == "TC-PAR-01-1"


def test_generated_sample_inherits_seed_labels():
    seed = get_sample("SEED-TC-01")
    s = build_sample(seed, {"intent": "new", "conflict_description": "A vs B"}, "contradictory", "TC-CF-01")
    assert s.expected_class == seed.expected_class and s.expected_conflicts == ["A vs B"]
    assert s.scenario == seed.scenario and s.expected_slots == {}
