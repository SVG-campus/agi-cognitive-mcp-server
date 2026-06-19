import os
import json
import pytest
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
    handle_run_compliance_audit
)

def test_cognitive_mcp_spec():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(current_dir, "mcp_spec.json")
    assert os.path.exists(spec_path)
    with open(spec_path, "r") as f:
        spec = json.load(f)
    assert "tools" in spec
    tool_names = [t["name"] for t in spec["tools"]]
    assert "sensorium_binning" in tool_names
    assert "causal_discovery" in tool_names
    assert "robustness_regularization" in tool_names
    assert "fetch_alpaca_market_quotes" in tool_names
    assert "read_system_file" in tool_names
    assert "search_project_directory" in tool_names
    assert "hchl_inference" in tool_names
    assert "run_compliance_audit" in tool_names

def test_robustness_regularization_handlers():
    # 1. Wasserstein DRO
    res_dro = handle_robustness_regularization_simulation({
        "simulation_type": "wasserstein_dro",
        "returns": [0.01, -0.02, 0.05, 0.04, -0.01],
        "radius": 0.02,
        "risk_aversion": 1.0
    })
    assert res_dro["dimensions"] == 1
    assert "robust_expected_return" in res_dro
    assert len(res_dro["optimal_weights"]) == 1

    # 2. Kolmogorov MDL
    res_mdl = handle_robustness_regularization_simulation({
        "simulation_type": "kolmogorov_mdl",
        "rule_complexities": [8, 16],
        "data_likelihoods": [0.5, 0.8]
    })
    assert res_mdl["num_candidates"] == 2
    assert res_mdl["optimal_candidate_index"] == 0

    # 3. RL Adjustment
    res_rl = handle_robustness_regularization_simulation({
        "simulation_type": "rl_adjustment",
        "reward_signals": [0.1, 0.2, -0.3, 0.4],
        "learning_rate": 0.1
    })
    assert res_rl["steps"] == 4
    assert len(res_rl["parameter_trajectory"]) == 4

def test_new_mcp_tools():
    # Test Alpaca Quotes
    res_alpaca = handle_fetch_alpaca_market_quotes({"symbol": "AAPL", "limit": 5})
    assert res_alpaca["symbol"] == "AAPL"
    assert len(res_alpaca["quotes"]) == 5
    assert "source" in res_alpaca

    # Test Search Project Directory
    res_search = handle_search_project_directory({"path": "agi-cognitive-agent-core", "pattern": "*.py"})
    assert res_search["pattern"] == "*.py"
    assert res_search["matches_count"] > 0
    assert any(m["name"] == "mcp_server.py" for m in res_search["results"])

    # Test Read System File
    current_test_dir = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(current_test_dir, "mcp_spec.json")
    res_read = handle_read_system_file({"path": spec_path})
    assert "tools" in res_read["content"]
    assert res_read["size_bytes"] > 0

    # Test security boundary rejection
    with pytest.raises(PermissionError):
        handle_read_system_file({"path": "../../../../../Windows/System32/drivers/etc/hosts"})

def test_hchl_mcp_handlers():
    # Test HCHL inference (safe query)
    res = handle_hchl_inference({
        "prompt": "Determine the first Betti number of a 2D torus.",
        "domain": "epistemic"
    })
    assert "response" in res
    assert "telemetry" in res
    assert res["telemetry"]["d_separated"]
    assert res["telemetry"]["betti_1_approx"] >= 0

    # Test HCHL compliance audit
    audit = handle_run_compliance_audit({})
    assert "standards_compliance" in audit
    assert "status" in audit
    assert audit["standards_compliance"]["ISO_IEC_42001"]["governance_score_pct"] >= 0.0
