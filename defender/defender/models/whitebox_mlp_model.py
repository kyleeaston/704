import json, os
from pathlib import Path
import numpy as np
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class WhiteboxMLPEmberModel:
    """
    Minimal adapter to match the template's model interface:
      - predict(bytez) -> int in {0,1}
      - model_info() -> dict

    We reuse the *existing* env-driven main. For minimal changes, the __init__
    signature mirrors StatefulNNEmberModel, but unused args are accepted.
    """
    def __init__(self,
                 model_gz_path: str,      # will actually point to our .pt
                 model_thresh: float = None,
                 model_ball_thresh: float = None,   # ignored (kept for compat)
                 model_max_history: int = None,     # ignored (kept for compat)
                 model_name: str = "whitebox_mlp"):

        # Resolve paths near the provided model path (works with existing env var)
        self.weights_pt = Path(model_gz_path)
        if not self.weights_pt.exists():
            # fallback to default location
            self.weights_pt = MODELS_DIR / "whitebox_mlp.pt"

        self.scaler_path   = MODELS_DIR / "whitebox_scaler.joblib"
        self.threshold_json= MODELS_DIR / "whitebox_threshold.json"
        self.meta_path     = MODELS_DIR / "whitebox_model_meta.json"

        # Threshold: env/param overrides JSON if provided
        self.threshold = float(model_thresh) if model_thresh is not None else 0.5
        if self.threshold_json.exists():
            try:
                obj = json.loads(self.threshold_json.read_text())
                self.threshold = float(obj.get("threshold", self.threshold))
            except Exception:
                pass

        # Load scaler (optional)
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

        # Load meta
        meta = {"input_dim": 2381, "hidden_dims": [512, 256]}
        if self.meta_path.exists():
            try:
                meta.update(json.loads(self.meta_path.read_text()))
            except Exception:
                pass
        self.input_dim  = int(meta["input_dim"])
        self.hidden     = list(meta["hidden_dims"])

        # Torch model
        import torch, torch.nn as nn
        class MLP(nn.Module):
            def __init__(self, d, hs):
                super().__init__()
                layers, last = [], d
                for h in hs:
                    layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.10)]
                    last = h
                layers += [nn.Linear(last, 1)]
                self.net = nn.Sequential(*layers)

            def forward(self, x): return self.net(x).squeeze(-1)


  

        self.torch = torch
        self.model = MLP(self.input_dim, self.hidden).eval()
        state = torch.load(self.weights_pt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            self.model.load_state_dict(state)

        # Feature extractor: EMBER-style (2,381-d) from bytes
        try:
            import ember, lief  # noqa: F401
            self._ember_ok = True
            import ember as _ember
            self.extractor = _ember.PEFeatureExtractor()
        except Exception:
            self._ember_ok = False
            self.extractor = None

    # ---- Interface expected by the app ----
    def predict(self, bytez: bytes) -> int:
        x = self._bytes_to_features(bytez).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        with self.torch.no_grad():
            t = self.torch.from_numpy(x.astype(np.float32))
            prob = float(self.torch.sigmoid(self.model(t)).cpu().numpy().ravel()[0])
        return int(prob >= self.threshold)

    def model_info(self) -> dict:
        return {
            "name": "whitebox_mlp",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden,
            "threshold": float(self.threshold),
            "has_scaler": bool(self.scaler is not None),
            "ember_extractor": bool(self._ember_ok),
            "weights": str(self.weights_pt.name),
        }

    # ---- Helpers ----
    def _bytes_to_features(self, bytez: bytes) -> np.ndarray:
        if self._ember_ok and self.extractor is not None:
            try:
                if hasattr(self.extractor, "feature_vector"):
                    vec = self.extractor.feature_vector(bytez)
                elif hasattr(self.extractor, "extract"):
                    vec = self.extractor.extract(bytez)
                else:
                    raise RuntimeError("Unknown EMBER extractor API")
                v = np.asarray(vec, dtype=np.float32)
                # pad/trim defensively
                if v.shape[0] != self.input_dim:
                    vv = np.zeros((self.input_dim,), dtype=np.float32)
                    n = min(self.input_dim, v.shape[0]); vv[:n] = v[:n]
                    v = vv
                return v
            except Exception:
                pass
        # fallback: zero vector (keeps API responsive)
        return np.zeros((self.input_dim,), dtype=np.float32)
