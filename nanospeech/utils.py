from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def fetch_from_hub(hf_repo: str, quantization_bits: Optional[int] = None) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.txt"],
        )
    )
    return model_path
