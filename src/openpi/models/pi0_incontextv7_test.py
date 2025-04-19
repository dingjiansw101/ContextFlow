import unittest
import flax.nnx as nnx
import openpi.shared.nnx_utils as nnx_utils


# ----------------------------------------------------------------------
# 1.  Atomic regex filters
# ----------------------------------------------------------------------
gemma_params_filter         = nnx_utils.PathRegex(r".*llm.*")                   # matches every “llm” weight
action_expert_params_filter = nnx_utils.PathRegex(r".*llm.*_1.*")              # action‑expert
prompt_expert_params_filter = nnx_utils.PathRegex(r".*llm.*_prompt_expert.*")  # prompt‑expert

# ----------------------------------------------------------------------
# 2.  Composite filter: Gemma ∧ ¬Action ∧ ¬Prompt
# ----------------------------------------------------------------------
gemma_only_filter = nnx.All(
    gemma_params_filter,
    nnx.Not(action_expert_params_filter),
    nnx.Not(prompt_expert_params_filter),
)

# ----------------------------------------------------------------------
# 3.  Test‑case definitions
# ----------------------------------------------------------------------
class TestGemmaOnlyFilter(unittest.TestCase):
    """Ensure gemma_only_filter selects the correct parameter paths."""

    def setUp(self):
        # path : should_match?
        self.samples = {
            # --- Gemma trunk (expected True) ---------------------------
            "model.llm.block_0.weight":                    True,
            "model.encoder.llm.attention.q_proj.bias":     True,

            # --- Action expert (expected False) -----------------------
            "model.llm_1.block_3.weight":                  False,

            # --- Prompt expert (expected False) -----------------------
            "model.llm_prompt_expert.block_2.weight":      False,

            # --- LoRA adapters anywhere (always False) ----------------
            "model.llm.block_0.lora_A.weight":             False,
            "model.llm_1.block_3.lora_A.weight":           False,
        }

    def test_filter_matches(self):
        for path, should_match in self.samples.items():
            with self.subTest(path=path):
                self.assertEqual(
                    gemma_only_filter(path),
                    should_match,
                    msg=f"Path '{path}' was expected to be {should_match}",
                )

if __name__ == "__main__":
    unittest.main()
