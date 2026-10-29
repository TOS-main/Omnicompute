import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os  # noqa: E402

# tests run GGUF models on the mesh code; the llama.cpp engine has its own tests (test_llamacpp.py)
os.environ.setdefault("OMNICOMPUTE_LLAMA", "off")
