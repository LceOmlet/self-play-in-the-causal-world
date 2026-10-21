"""Verify source pinning and exact reuse of official comparison semantics."""
import unittest
from types import SimpleNamespace as NS
from official_vllm_validation import official_rules, rounded_result


class OfficialRules(unittest.TestCase):
    def test_source_pinned_compile_rule_is_rounding_not_absolute_epsilon(self):
        helper,_ = official_rules()
        a = rounded_result(helper,{'a':-.101,'b':-1.201})
        b = rounded_result(helper,{'a':-.104,'b':-1.204})
        self.assertEqual(a,b)
        # A tiny change crossing a rounding boundary must fail. No noise-based
        # allowance or absolute 0.01 inequality is substituted for upstream.
        c = rounded_result(helper,{'a':-.106,'b':-1.204})
        self.assertNotEqual(b,c)
        self.assertNotEqual(a,rounded_result(helper,{'different_token':-.101,'b':-1.201}))

    def test_official_graph_assertion_compares_full_text_after_lowercasing(self):
        _,rule = official_rules()
        def check(a,b):
            return eval(rule,{}, {'piecewise_res':NS(outputs=[NS(text=a)]),
                                  'full_res':NS(outputs=[NS(text=b)])})
        self.assertTrue(check('Hello\nworld','hello\nWORLD'))
        self.assertFalse(check('Hello\nworld','Hello world'))
        self.assertFalse(check('shared prefix x','shared prefix y'))


if __name__ == '__main__':
    unittest.main()
