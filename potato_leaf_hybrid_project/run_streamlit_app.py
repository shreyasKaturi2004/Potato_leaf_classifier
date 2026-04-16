from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
APP_PATH = PROJECT_ROOT / "src" / "app.py"


def main() -> None:
    if VENDOR_DIR.exists():
        sys.path.append(str(VENDOR_DIR))

    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(APP_PATH),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
