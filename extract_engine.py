#!/usr/bin/env python3
"""
extract_engine.py — regenerate wlsweep/*.py from the notebook.

The notebook stores its four engine modules as raw string literals
(_SRC_NSE_DATA, _SRC_SWEEP_ENGINE, _SRC_SCREENER, _SRC_TESTS_FIXTURES) and
writes them to wlsweep/ at runtime. This script does exactly the same thing
offline, so the automated run executes byte-identical code.

Run it whenever you update the notebook:

    python extract_engine.py && python -m pytest wlsweep/tests_fixtures.py -q
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebook", "NSE_Weekly_Liquidity_Sweep_Screener.ipynb")
PKG = os.path.join(ROOT, "wlsweep")

# cell index in the notebook -> output module name
MODULES = {
    "_SRC_NSE_DATA": "nse_data.py",
    "_SRC_SWEEP_ENGINE": "sweep_engine.py",
    "_SRC_SCREENER": "screener.py",
    "_SRC_TESTS_FIXTURES": "tests_fixtures.py",
}


def main() -> int:
    if not os.path.exists(NB):
        print(f"notebook not found: {NB}", file=sys.stderr)
        return 1
    nb = json.load(open(NB, encoding="utf-8"))
    os.makedirs(PKG, exist_ok=True)

    found = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        for var, fname in MODULES.items():
            m = re.search(rf"^{var} = r'''\n(.*)\n'''\nwith open", src, re.S | re.M)
            if not m:
                continue
            body = m.group(1)
            try:
                ast.parse(body)
            except SyntaxError as exc:
                print(f"! {fname}: syntax error after extraction: {exc}", file=sys.stderr)
                return 2
            path = os.path.join(PKG, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            print(f"{fname:22s} {len(body):,} bytes")
            found += 1

    if found != len(MODULES):
        print(f"! expected {len(MODULES)} modules, extracted {found}", file=sys.stderr)
        return 3
    print("ok — now run:  python -m pytest wlsweep/tests_fixtures.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
