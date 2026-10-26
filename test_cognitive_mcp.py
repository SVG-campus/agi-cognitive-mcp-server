import os
import json
import unittest
import numpy as np
from mcp_server import (
    handle_sensorium_binning,
    handle_causal_discovery,
    handle_epistemic_gp_update,
    handle_thermodynamic_audit,
    handle_meta_learning_lora,
    handle_robustness_regularization_simulation,
    handle_fetch_alpaca_market_quotes,
    handle_read_system_file,
    handle_search_project_directory,
    handle_hchl_inference,
    handle_run_compliance_audit,
    handle_mera_holographic_consensus,
    handle_mera_hierarchical_consensus,
    handle_betti_homology_evaluator,
    handle_bootstrap_confidence_interval,
    handle_null_y_shuffle_test
)

class TestAgiCognitiveMcp(unittest.TestCase):
    def test_cognitive_mcp_spec(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spec_path = os.path.join(current_dir, "mcp_spec.json")
        self.assertTrue(os.path.exists(spec_path))
        with open(spec_path, "r") as f:
            spec = json.load(f)
        self.assertIn("tools", spec)
        tool_names = [t["name"] for t in spec["tools"]]
        self.assertIn("sensorium_binning", tool_names)
        self.assertIn("causal_discovery", tool_names)
        self.assertIn("robustness_regularization", tool_names)
        self.assertIn("fetch_alpaca_market_quotes", tool_names)
        self.assertIn("read_system_file", tool_names)
        self.assertIn("search_project_directory", tool_names)
        self.assertIn("hchl_inference", tool_names)
        self.assertIn("run_compliance_audit", tool_names)
        self.assertIn("mera_holographic_consensus", tool_names)
        self.assertIn("mera_hierarchical_consensus", tool_names)
        self.assertIn("betti_homology_evaluator", tool_names)

    def test_robustness_regularization_handlers(self):
        # 1. Wasserstein DRO
        res_dro = handle_robustness_regularization_simulation({
            "simulation_type": "wasserstein_dro",
            "returns": [0.01, -0.02, 0.05, 0.04, -0.01],
            "radius": 0.02,
            "risk_aversion": 1.0
        })
        self.assertEqual(res_dro["dimensions"], 1)
        self.assertIn("robust_expected_return", res_dro)
        self.assertEqual(len(res_dro["optimal_weights"]), 1)

        # 2. Kolmogorov MDL
        res_mdl = handle_robustness_regularization_simulation({
            "simulation_type": "kolmogorov_mdl",
            "rule_complexities": [8, 16],
            "data_likelihoods": [0.5, 0.8]
        })
        self.assertEqual(res_mdl["num_candidates"], 2)
        self.assertEqual(res_mdl["optimal_candidate_index"], 0)

        # 3. RL Adjustment
        res_rl = handle_robustness_regularization_simulation({
            "simulation_type": "rl_adjustment",
            "reward_signals": [0.1, 0.2, -0.3, 0.4],
            "learning_rate": 0.1
        })
        self.assertEqual(res_rl["steps"], 4)
        self.assertEqual(len(res_rl["parameter_trajectory"]), 4)

    def test_new_mcp_tools(self):
        res_alpaca = handle_fetch_alpaca_market_quotes({"symbol": "AAPL", "limit": 5})
        self.assertEqual(res_alpaca["symbol"], "AAPL")
        self.assertEqual(len(res_alpaca["quotes"]), 5)

        res_search = handle_search_project_directory({"path": "agi-cognitive-agent-core", "pattern": "*.py"})
        self.assertEqual(res_search["pattern"], "*.py")
        self.assertTrue(res_search["matches_count"] > 0)

        current_test_dir = os.path.dirname(os.path.abspath(__file__))
        spec_path = os.path.join(current_test_dir, "mcp_spec.json")
        res_read = handle_read_system_file({"path": spec_path})
        self.assertIn("tools", res_read["content"])

        with self.assertRaises(PermissionError):
            handle_read_system_file({"path": "../../../../../Windows/System32/drivers/etc/hosts"})

    def test_hchl_mcp_handlers(self):
        res = handle_hchl_inference({
            "prompt": "Determine the first Betti number of a 2D torus.",
            "domain": "epistemic"
        })
        self.assertIn("response", res)
        self.assertIn("telemetry", res)

        audit = handle_run_compliance_audit({})
        self.assertIn("standards_compliance", audit)

    def test_cross_repo_mera_and_research_tools(self):
        # MERA Consensus
        res_mera = handle_mera_holographic_consensus({
            "belief_vectors": [[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]],
            "expected_returns": [100.0, 10.0]
        })
        self.assertIn("winning_index", res_mera)

        # Bootstrap CI
        res_boot = handle_bootstrap_confidence_interval({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "n_boot": 100
        })
        self.assertIn("mean_estimate", res_boot)
        self.assertIn("ci_lower", res_boot)

        # Null Y-shuffle test
        res_null = handle_null_y_shuffle_test({
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "x": [2, 4, 6, 8, 10, 12, 14, 16],
            "n_perm": 99
        })
        self.assertIn("permutation_p_value", res_null)

if __name__ == '__main__':
    unittest.main()
