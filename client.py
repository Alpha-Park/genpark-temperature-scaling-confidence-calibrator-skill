"""
Temperature Scaling Confidence Calibrator Skill Client
Pure Python Standard Library implementation of post-hoc temperature scaling (Guo et al.).
Optimizes a single temperature parameter T > 0 on a validation set using negative log-likelihood
and evaluates Expected Calibration Error (ECE) and Brier Score.
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class TemperatureCalibrator:
    """
    Calibrator for overconfident or underconfident neural network / LLM logits.
    Uses golden section search to find optimal T that minimizes validation NLL.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, temperature)

    @staticmethod
    def _softmax(logits: List[float], temp: float) -> List[float]:
        scaled = [l / temp for l in logits]
        max_val = max(scaled)
        exp_vals = [math.exp(v - max_val) for v in scaled]
        sum_exp = sum(exp_vals)
        return [v / sum_exp for v in exp_vals]

    def compute_nll(self, validation_logits: List[List[float]], labels: List[int], temp: float) -> float:
        """Compute Negative Log-Likelihood given temperature."""
        total_nll = 0.0
        for logits, label in zip(validation_logits, labels):
            probs = self._softmax(logits, temp)
            p_true = max(1e-12, probs[label])
            total_nll -= math.log(p_true)
        return total_nll / len(labels)

    def fit(self, validation_logits: List[List[float]], labels: List[int], search_min: float = 0.1, search_max: float = 5.0) -> float:
        """
        Fit optimal temperature T using Golden Section Search.
        """
        if not validation_logits or len(validation_logits) != len(labels):
            raise ValueError("Invalid validation logits and labels")

        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi

        a = search_min
        b = search_max
        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

        fc = self.compute_nll(validation_logits, labels, c)
        fd = self.compute_nll(validation_logits, labels, d)

        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = a + resphi * (b - a)
                fc = self.compute_nll(validation_logits, labels, c)
            else:
                a = c
                c = d
                fc = fd
                d = b - resphi * (b - a)
                fd = self.compute_nll(validation_logits, labels, d)

        self.temperature = (a + b) / 2.0
        return self.temperature

    def calibrate_logits(self, logits: List[float]) -> List[float]:
        """Calibrate a single logit vector into calibrated probabilities."""
        return self._softmax(logits, self.temperature)

    def evaluate_ece(self, logits_list: List[List[float]], labels: List[int], num_bins: int = 10) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) and Brier Score.
        """
        n = len(labels)
        if n == 0:
            return {"ece": 0.0, "brier_score": 0.0}

        confidences = []
        accuracies = []
        brier_sum = 0.0

        for logits, label in zip(logits_list, labels):
            probs = self.calibrate_logits(logits)
            max_prob = max(probs)
            pred_label = probs.index(max_prob)
            confidences.append(max_prob)
            accuracies.append(1 if pred_label == label else 0)

            # Brier Score = (1/K) sum((p_k - y_k)^2)
            for k in range(len(logits)):
                y_k = 1.0 if k == label else 0.0
                brier_sum += (probs[k] - y_k) ** 2

        brier_score = brier_sum / (n * len(logits_list[0]))

        # Binning for ECE
        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        ece = 0.0

        for b in range(num_bins):
            low = bin_boundaries[b]
            high = bin_boundaries[b + 1]

            bin_indices = [
                i for i, conf in enumerate(confidences)
                if (low <= conf < high) or (b == num_bins - 1 and low <= conf <= high)
            ]

            if bin_indices:
                bin_acc = sum(accuracies[i] for i in bin_indices) / len(bin_indices)
                bin_conf = sum(confidences[i] for i in bin_indices) / len(bin_indices)
                ece += (len(bin_indices) / n) * abs(bin_acc - bin_conf)

        return {
            "ece": ece,
            "brier_score": brier_score,
            "temperature": self.temperature,
            "samples": n
        }
