from adapters.llm.scripted import ScriptedLanguageModel, first_valid_value
from adapters.users import ScriptedUser, SimulatedUser
from core.ports import LanguageModel, UserChannel, UserSimulationModel
from models.dialogue import Question
from models.ontology import RequiresParameter


def test_adapters_satisfy_the_ports():
    llm = ScriptedLanguageModel()
    assert isinstance(llm, LanguageModel) and isinstance(llm, UserSimulationModel)
    assert isinstance(ScriptedUser(), UserChannel)
    assert isinstance(SimulatedUser(llm, "persona"), UserChannel)


def test_simulated_user_delegates_to_the_model():
    class Echo:
        def simulate_user(self, scenario, history, question):
            return f"{scenario}|{len(history)}|{question}"
    user = SimulatedUser(Echo(), "port manager")
    assert user.ask(Question("Cost?"), [{"role": "assistant", "content": "q"}]) == "port manager|1|Cost?"


def test_scripted_user_answers_in_order_then_default():
    user = ScriptedUser(["a", "b"], default="z")
    assert [user.ask(Question(str(i)), []) for i in range(3)] == ["a", "b", "z"]
    assert [q.text for q in user.questions] == ["0", "1", "2"]


def test_scripted_user_accepts_a_function():
    user = ScriptedUser(lambda q: q.text.upper())
    assert user.ask(Question("hi"), []) == "HI"


def test_first_valid_value():
    assert first_valid_value(RequiresParameter("x", "enum", "?", enum_values=["A", "B"])) == "A"
    assert first_valid_value(RequiresParameter("x", "bool", "?")) is False
