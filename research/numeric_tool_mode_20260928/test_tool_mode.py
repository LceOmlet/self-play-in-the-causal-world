"""Boundary tests only; no substitute for real NsJail/Jupyter validation."""
import inspect
import json
import unittest
from types import SimpleNamespace

from tool_mode import JupyterNumericSession, make_numeric_environment


class CausalOwnerFixture:
    def __init__(self):
        self._episode = None
        self.reward = 0.0

    def reset(self, *, prompt, **kwargs):
        self._episode = SimpleNamespace(completed=False, private_truth="NOT_VISIBLE")
        self.private_row = kwargs
        return "Use act."

    def act(self, command: dict[str, object]) -> str:
        if command.get("terminal"):
            self._episode.completed = True
            self.reward = 0.75
            return "Terminal accepted."
        return "Observed counts: [2, 3]"

    def get_reward(self):
        return self.reward

    @property
    def episode(self):
        return self._episode


class SessionFixture:
    def __init__(self):
        self.closed = False
        self.calls = []

    def close(self):
        self.closed = True

    def execute(self, code, public_context):
        self.calls.append((code, public_context))
        return "numeric fixture result"


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []

        def factory():
            session = SessionFixture()
            self.sessions.append(session)
            return session

        self.env_type = make_numeric_environment(CausalOwnerFixture, factory)
        self.env = self.env_type()
        self.env.reset(prompt="PUBLIC_TASK", sample_index=999, terminal_truth_json="SECRET")

    def test_only_two_tools_under_official_discovery_rule(self):
        names = {name for name, _ in inspect.getmembers(self.env, predicate=inspect.ismethod)
                 if not name.startswith("_") and name not in {"reset", "get_reward"}}
        self.assertEqual(names, {"act", "numerics"})

    def test_hidden_row_and_truth_never_cross_numeric_boundary(self):
        self.assertEqual(self.env.act({"observe": True}), "Observed counts: [2, 3]")
        self.env.numerics("print(public)")
        payload = self.sessions[0].calls[0][1]
        self.assertEqual(payload, {"prompt": "PUBLIC_TASK", "observations": [
            {"command": {"observe": True}, "feedback": "Observed counts: [2, 3]"}]})

    def test_reuse_within_answer_reset_discards_state(self):
        self.env.numerics("x = 1")
        self.env.numerics("print(x)")
        self.assertEqual(len(self.sessions), 1)
        self.env.reset(prompt="NEXT")
        self.assertTrue(self.sessions[0].closed)
        self.env.numerics("print('new')")
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual(self.sessions[1].calls[0][1]["observations"], [])

    def test_four_answers_get_four_separate_sessions(self):
        answers = [self.env_type() for _ in range(4)]
        for answer in answers:
            answer.reset(prompt="same public question")
            answer.numerics("x = 1")
        self.assertEqual(len({id(a._numeric_session) for a in answers}), 4)

    def test_terminal_closes_compute_and_keeps_owner_reward(self):
        self.env.numerics("x = 1")
        self.assertEqual(self.env.act({"terminal": True}), "Terminal accepted.")
        self.assertTrue(self.sessions[0].closed)
        self.assertEqual(self.env.get_reward(), 0.75)
        self.env.numerics("print('must not execute')")
        self.assertEqual(len(self.sessions[0].calls), 1)

    def test_reward_collection_closes_a_truncated_answer(self):
        self.env.numerics("x = 1")
        self.assertEqual(self.env.get_reward(), 0.0)
        self.assertTrue(self.sessions[0].closed)

    def test_extra_public_method_is_rejected(self):
        class LeakyOwner(CausalOwnerFixture):
            def debug_truth(self):
                return "secret"
        with self.assertRaises(ValueError):
            make_numeric_environment(LeakyOwner, SessionFixture)

    def test_shared_factory_is_rejected(self):
        shared = SessionFixture()
        cls = make_numeric_environment(CausalOwnerFixture, lambda: shared)
        left, right = cls(), cls()
        left.reset(prompt="left")
        right.reset(prompt="right")
        left.numerics("x = 1")
        with self.assertRaises(RuntimeError):
            right.numerics("print(x)")
        self.assertFalse(shared.closed)

    def test_jupyter_timeout_destroys_session_without_retry(self):
        destroyed = []
        class Client:
            def execute_interactive(self, **kwargs):
                self.kwargs = kwargs
                raise TimeoutError("PRIVATE_HOST_PATH")
        client = Client()
        session = JupyterNumericSession(client, lambda: destroyed.append(True))
        result = json.loads(session.execute("print(1)", {"prompt": "public"}))
        self.assertTrue(result["state_lost"])
        self.assertEqual(destroyed, [True])
        self.assertEqual(client.kwargs["timeout"], 30)
        self.assertFalse(client.kwargs["allow_stdin"])
        self.assertNotIn("PRIVATE_HOST_PATH", json.dumps(result))

    def test_output_limit_counts_utf8_bytes_and_reports_failure(self):
        class Client:
            def execute_interactive(self, **kwargs):
                kwargs["output_hook"]({"header": {"msg_type": "stream"},
                                       "content": {"text": "数" * 3000}})
        session = JupyterNumericSession(Client(), lambda: None)
        result = json.loads(session.execute("unused fixture", {}))
        self.assertEqual(result["error"], "output_limit")
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
