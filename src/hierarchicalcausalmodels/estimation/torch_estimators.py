"""Torch-based per-unit regressors for ATE estimation."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch


class MLPRegressorPerUnit:
    """Minimal MLP regressor for (A, Y) per unit; .fit(X, y), .predict(X) compatible with sklearn-style usage."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        max_epochs: int = 50,
        hidden_sizes: tuple[int, ...] = (8,),
        lr: float = 1e-2,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self._model: Optional[torch.nn.Module] = None
        self._fitted = False

    def _build_model(self, input_size: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        prev = input_size
        for h in self.hidden_sizes:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressorPerUnit":
        """Fit MLP on (X, y); X shape (n, 1) or (n,)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        in_size = X.shape[1]
        self._model = self._build_model(in_size)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        Xt = torch.from_numpy(X).to(self.device)
        yt = torch.from_numpy(y).reshape(-1, 1).to(self.device)
        self._model.train()
        for _ in range(self.max_epochs):
            opt.zero_grad()
            out = self._model(Xt)
            loss = ((out - yt) ** 2).mean()
            loss.backward()
            opt.step()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict; X shape (n, 1) or (n,)."""
        if not self._fitted or self._model is None:
            raise RuntimeError("MLPRegressorPerUnit not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(self.device)
            out = self._model(Xt)
        return out.cpu().numpy().ravel()
