#!/usr/bin/env python3
"""Compatibility launcher that delegates to the MINT auto-review loop."""

from __future__ import annotations

from mint_auto_review_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
