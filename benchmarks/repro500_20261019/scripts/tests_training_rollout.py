"""Exercise the actual installed training methods with synthetic generations only."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import make_backend, make_trainer, read, training_processor, TRAIN


class Environment:
    def __init__(self, result='2'):
        self.ops = []
        self.result = result
        self.dispatch = {'act':self.act,'numerics':self.numerics}

    def act(self, command: dict) -> str:
        """Submit a synthetic answer.

        Args:
            command: Synthetic command object.
        """
        self.ops.append('act')
        return 'Episode complete; finish with an acknowledgement.'

    def numerics(self, code: str) -> str:
        """Run a synthetic computation.

        Args:
            code: Synthetic Python source.
        """
        self.ops.append('numerics')
        return self.result


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.llm_engine = NS(model_config=NS(max_model_len=32768))

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert sampling_params.max_tokens == 31497
        assert sampling_params.temperature == 1.0
        assert sampling_params.top_p == 1.0
        assert sampling_params.top_k == 0
        index = len(self.calls)
        self.calls.append((prompts,sampling_params))
        ids = self.responses[index]
        return [NS(prompt_token_ids=p['prompt_token_ids'], outputs=[NS(token_ids=ids,
            logprobs=[{token:NS(logprob=-0.1,rank=1)} for token in ids])]) for p in prompts]


class TrainingRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processing,cls.tokenizer,cls.template = training_processor()
        cls.config = read(TRAIN/'configuration.json')

    def tokens(self, assistant, env):
        from transformers.utils import get_json_schema
        prompt = [{'role':'user','content':'Synthetic generation protocol test only.'}]
        kwargs = dict(tools=[get_json_schema(env.act),get_json_schema(env.numerics)],
            chat_template=self.template,tokenize=True,return_dict=True,enable_thinking=False)
        prefix = self.processing.apply_chat_template([prompt],add_generation_prompt=True,**kwargs)['input_ids'][0]
        full = self.processing.apply_chat_template([prompt+[assistant]],add_generation_prompt=False,**kwargs)['input_ids'][0]
        self.assertEqual(full[:len(prefix)],prefix)
        ids = full[len(prefix):]
        if self.tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(self.tokenizer.eos_token_id)+1]
        return ids

    def tool(self, name, arguments):
        return {'role':'assistant','content':'','tool_calls':[
            {'type':'function','function':{'name':name,'arguments':arguments}}]}

    def fixture(self, assistants, env=None):
        env = env or Environment()
        responses = [self.tokens(a,env) for a in assistants]
        llm = FakeLLM(responses)
        backend = make_backend(llm,self.config)
        trainer = make_trainer(self.processing,self.tokenizer,self.template,backend,[env],self.config)
        return trainer,llm,env

    def generate(self, trainer):
        return trainer._generate([[{'role':'user','content':'Synthetic generation protocol test only.'}]])

    def test_no_8192_cap(self):
        assistant = {'role':'assistant','content':' synthetic'*9000}
        trainer,llm,env = self.fixture([assistant])
        output = self.generate(trainer)
        self.assertGreater(len(output[1][0]),8192)
        self.assertEqual(len(llm.calls),1)
        self.assertFalse(env.ops)

    def test_tools_continue_then_terminal_acknowledgement(self):
        trainer,llm,env = self.fixture([
            self.tool('numerics',{'code':'print(2)'}),
            self.tool('act',{'command':{'type':'answer','synthetic':True}}),
            {'role':'assistant','content':'Completed.'}])
        result = self.generate(trainer)
        self.assertEqual(env.ops,['numerics','act'])
        self.assertEqual(len(llm.calls),3)
        self.assertEqual(sum(mask == 0 for mask in result[2][0]) > 0,True)
        self.assertIn('Completed.',str(result[3][0][-1]))

    def test_unknown_tool_feedback_does_not_abort(self):
        trainer,llm,env = self.fixture([
            self.tool('numerics',{'code':'print(2)'}),
            self.tool('act',{'command':{'type':'answer'}}),
            {'role':'assistant','content':'Completed.'}])
        trainer._sync_tool_dicts[0].pop('numerics')
        result = self.generate(trainer)
        self.assertEqual(env.ops,['act'])
        self.assertEqual(len(llm.calls),3)
        self.assertIn('Tool numerics not found',str(result[3]))

    def test_overlong_tool_feedback_uses_training_rollback(self):
        env = Environment(' synthetic'*1000)
        assistants = [self.tool('numerics',{'code':'print(2)'})]
        trainer,llm,env = self.fixture(assistants,env)
        trainer.max_completion_length = len(llm.responses[0])+5
        result = self.generate(trainer)
        self.assertEqual(env.ops,['numerics'])
        self.assertEqual(len(llm.calls),1)
        self.assertEqual(result[1][0],llm.responses[0])
        self.assertFalse(any(m['role']=='tool' for m in result[3][0]))

    def test_post_tool_generation_is_clipped_to_total_budget(self):
        trainer,llm,env = self.fixture([
            self.tool('numerics',{'code':'print(2)'}),
            {'role':'assistant','content':' synthetic'*1000}])
        # Include the real training tool suffix in the small synthetic budget.
        suffix = trainer._get_tool_suffix_ids([{'role':'tool','name':'numerics','content':'2'}])
        budget = len(llm.responses[0])+len(suffix)+5
        trainer.max_completion_length = budget
        result = self.generate(trainer)
        self.assertEqual(len(llm.calls),2)
        self.assertEqual(len(result[1][0]),budget)
        self.assertEqual(len(result[2][0]),budget)


if __name__ == '__main__':
    unittest.main()
