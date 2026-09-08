"""Exercise diagnostic admission/cancellation without a trainer or GPU mock."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from borrow_vllm import borrowed_turn


class BorrowingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "prepared.json").write_text(json.dumps({"step": 50}))
        self.states = {"training-1": object()}
        self.admitted = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = False

        async def is_sleeping():
            return False

        async def generate(**kwargs):
            self.admitted.set()
            self.states[kwargs["request_id"]] = object()
            try:
                await self.finish.wait()
                return SimpleNamespace(stop_reason="stop")
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                self.states.pop(kwargs["request_id"])

        self.server = SimpleNamespace(
            _submission_paused=False,
            config=SimpleNamespace(max_num_seqs=4),
            engine=SimpleNamespace(
                is_sleeping=is_sleeping,
                output_processor=SimpleNamespace(request_states=self.states),
            ),
            generate=generate,
        )

        native = self.server

        async def inspect(function, run_path):
            return function(native, run_path)

        self.server = SimpleNamespace(
            __ray_call__=SimpleNamespace(remote=inspect),
            generate=SimpleNamespace(
                remote=lambda **kwargs: asyncio.create_task(generate(**kwargs))
            ),
        )

    def call(self, step=50):
        return borrowed_turn(
            self.server,
            run_path=str(self.root),
            request_id="cpt-diagnostic-test",
            expected_step=step,
            cancel_request=lambda reference: reference.cancel(),
        )

    async def test_idle_or_full_server_never_admits(self):
        for count in [0, 4]:
            self.states.clear()
            self.states.update({f"training-{i}": object() for i in range(count)})
            self.assertEqual((await self.call())["status"], "yielded")
            self.assertFalse(self.admitted.is_set())

    async def test_training_window_closes_cancels_only_diagnostic(self):
        task = asyncio.create_task(self.call())
        await self.admitted.wait()
        self.states.pop("training-1")
        self.assertEqual((await task)["status"], "yielded")
        self.assertTrue(self.cancelled)
        self.assertFalse(self.states)

    async def test_training_takes_capacity_cancels_only_diagnostic(self):
        task = asyncio.create_task(self.call())
        await self.admitted.wait()
        others = {f"training-{i}": object() for i in range(1, 5)}
        self.states.update(others)
        self.assertEqual((await task)["status"], "yielded")
        self.assertTrue(self.cancelled)
        self.assertEqual(self.states, others)

    async def test_checkpoint_change_does_not_admit(self):
        self.assertEqual((await self.call(step=49))["status"], "version_changed")
        self.assertFalse(self.admitted.is_set())

    async def test_complete_generation_preserves_training_request(self):
        task = asyncio.create_task(self.call())
        await self.admitted.wait()
        self.finish.set()
        self.assertEqual((await task)["status"], "complete")
        self.assertFalse(self.cancelled)
        self.assertEqual(set(self.states), {"training-1"})


if __name__ == "__main__":
    unittest.main()
