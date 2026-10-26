"""Build selected checkpoint's indexes explicitly; never start a sweep on server startup."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from model_config import CACHE, STATE, GGUF
from runtime import Runtime
if GGUF:
    from gguf_runtime import GGUFRuntime as Runtime

p = argparse.ArgumentParser()
p.add_argument("--lens", action="store_true")
p.add_argument("--scale", action="store_true")
args = p.parse_args()
rt = Runtime(str(STATE / "stats"))
if args.scale:
    rt.ensure_act_scale()
if args.lens:
    rt.build_lens_index(str(CACHE / "lens_index.npz"), chunk=64)
print(CACHE)
