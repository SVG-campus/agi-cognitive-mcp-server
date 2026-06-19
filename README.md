# AGI Cognitive Model Context Protocol (MCP) Server

[![Model Context Protocol](https://img.shields.io/badge/MCP-Protocol-blue.svg)](https://modelcontextprotocol.io/)
[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.14-green.svg)](https://python.org)

An enterprise-ready **Model Context Protocol (MCP)** server providing programmatic access to causal discovery, persistent homology ($B_1$ topological calculations), epistemic Gaussian Process updates, Landauer thermodynamic limits, and local HCHL (Hypergeneralized Causal-Homological Latent) inference.

Researchers and agents can mount this server directly to run complex quantitative and topological simulations locally or in the cloud.

---

## 🛠️ Features

* **Causal Discovery**: Programmatic PC and FCI causal structure recovery algorithms.
* **Topological Data Analysis (TDA)**: 1-skeleton persistent homology cycle detection and filtration.
* **Epistemic GP Belief Updating**: Gaussian Process regression tracking for model beliefs and uncertainty prediction.
* **Thermodynamic Auditing**: Automated Landauer heat dissipation limit and computational complexity risk indicators.
* **Quantized LoRA Tuning**: Meta-learning matrices simulation with INT8 scaling factors.
* **Wasserstein DRO & MDL**: Robust optimization simulations under uncertainty radius $\epsilon$ and Kolmogorov AST code model complexity.
* **HCHL Core Inference**: Direct stdio pipeline calling the hypergeneralized local transformer.
* **ISO/IEC & NIST Auditing**: Real-time regulatory standard scoring compliance outputs.

---

## 🚀 Setup & Installation

### 1. Requirements
Ensure you have Python 3.12+ (or 3.14+) installed. Clone the repository and install requirements:

```bash
pip install -r requirements.txt
```

### 2. Sibling Dependency Note
This server acts as a gateway interface. It automatically detects and binds to parent/sibling submodules inside the main `compute-intelligence-orchestrator` project structure (e.g. `agi-cognitive-agent-core`, `automated-artificial-general-intelligence`, `post-exotic-research-compendium`).

If running standalone, ensure these sibling directories are available in your path or set `PYTHONPATH`:
```bash
export PYTHONPATH="/path/to/compute-intelligence-orchestrator/agi-cognitive-agent-core:/path/to/compute-intelligence-orchestrator/automated-artificial-general-intelligence"
```

---

## 🔌 Connection Setup

Add the following configuration blocks to connect this server to your preferred LLM host client.

### Claude Desktop Integration
Modify your `claude_desktop_config.json` (typically located in `%APPDATA%/Claude/claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agi-cognitive-mcp-server": {
      "command": "python",
      "args": [
        "c:/Users/svillalobosgonzalez1/Documents/GitHub/compute-intelligence-orchestrator/agi-cognitive-mcp-server/mcp_server.py"
      ],
      "env": {
        "APCA_API_KEY_ID": "your_alpaca_key_id",
        "APCA_API_SECRET_KEY": "your_alpaca_secret_key"
      }
    }
  }
}
```

### Cursor Integration
1. Go to **Cursor Settings** -> **Features** -> **MCP**.
2. Click **+ Add New MCP Server**.
3. Fill in details:
   - **Name**: `agi-cognitive-mcp-server`
   - **Type**: `stdio`
   - **Command**: `python c:/Users/svillalobosgonzalez1/Documents/GitHub/compute-intelligence-orchestrator/agi-cognitive-mcp-server/mcp_server.py`

---

## 🧪 Verification

You can verify that the server is working and resolving correctly by running the test suite:

```bash
pytest test_cognitive_mcp.py
```
