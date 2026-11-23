#!/usr/bin/env python3
"""Assemble the Hermes provider plugin directory from src/antigravity_acp_mcp.

Usage: python3 scripts/build_plugin.py <out-dir>
The package modules are copied as-is; the plugin's own __init__.py only imports _plugin_entry (which registers
the provider), so tests can import the package without Hermes's ``providers`` module.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "antigravity_acp_mcp"
MANIFEST = """name: antigravity-acp-mcp-provider
kind: model-provider
version: 0.1.0
description: Google Antigravity via the official ACP server; Hermes tools over a parked loopback MCP endpoint
author: Nanako0129
requires_hermes: ">=0.21.6"
provides_tools: []
provides_hooks: []
provides_middleware: []
requires_env: []
python_dependencies: []
"""


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src in sorted(PKG.glob("*.py")):
        if src.name != "__init__.py":
            shutil.copy2(src, out / src.name)
    (out / "__init__.py").write_text('"""Hermes provider plugin: Antigravity ACP with Hermes tools over MCP."""\n\n'
                                     "from . import _plugin_entry  # noqa: F401  (registers the provider)\n")
    (out / "plugin.yaml").write_text(MANIFEST)
    print(out)


main()
