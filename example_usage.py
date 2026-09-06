"""
Example usage of Temperature Scaling Confidence Calibrator Skill.
"""

from client import TemperatureCalibrator


def main():
    print("=== Temperature Scaling Confidence Calibrator Demonstration ===")

    # Simulated overconfident logits: 3 classes
    val_logits = [
        [5.2, 0.1, -1.0],  # true class 0 (overconfident)
        [0.2, 4.8, 1.1],   # true class 1 (overconfident)
        [3.1, 2.9, 0.5],   # true class 0 (slight margin)
        [1.0, 5.0, 0.2],   # true class 1
        [0.1, 1.2, 4.9]    # true class 2
    ]
    val_labels = [0, 1, 0, 1, 2]

    # Initialize uncalibrated (T = 1.0)
    calibrator = TemperatureCalibrator(temperature=1.0)
    uncal_ece = calibrator.evaluate_ece(val_logits, val_labels)
    print("Uncalibrated (T=1.0) ECE:", round(uncal_ece["ece"], 4))
    print("Uncalibrated Brier Score:", round(uncal_ece["brier_score"], 4))

    # Fit optimal temperature
    opt_temp = calibrator.fit(val_logits, val_labels)
    print("\nOptimized Temperature T*:", round(opt_temp, 4))

    cal_ece = calibrator.evaluate_ece(val_logits, val_labels)
    print("Calibrated ECE:", round(cal_ece["ece"], 4))
    print("Calibrated Brier Score:", round(cal_ece["brier_score"], 4))

    # Test sample calibration
    sample_logits = [4.5, 1.0, 0.5]
    print("\nSample Logits:", sample_logits)
    print("Calibrated Probs:", [round(p, 4) for p in calibrator.calibrate_logits(sample_logits)])


if __name__ == "__main__":
    main()
