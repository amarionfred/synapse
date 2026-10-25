"""The connection pipeline must accept a non-Qwen, unsharded local checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from test_adapters import model, tokenizer


def test_local_llama_connections(tmp_path):
    checkpoint = tmp_path/"checkpoint"
    model("Llama").save_pretrained(checkpoint)
    tokenizer().save_pretrained(checkpoint)
    output = tmp_path/"connections.npz"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, EXPLORER_NO_MODEL="0", EXPLORER_BUNDLED_CACHE="0")
    result = subprocess.run([sys.executable, str(root/"explorer/build_connections.py"), "--model", str(checkpoint),
                             "--k", "4", "--out", str(output)], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    z = np.load(output)
    assert z["down_n"].shape == (2, 48, 4)
    assert z["down_layer"].dtype == np.int32
    assert np.all(z["down_layer"][0] == 1)
    assert np.all(z["up_layer"][1] == 0)
    assert np.all(np.isfinite(z["down_z"]))
    assert np.all((z["down_n"] >= 0) & (z["down_n"] < 48))
