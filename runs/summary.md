# Drowsiness detection: test-set results

Dataset: https://huggingface.co/datasets/n7i5x9/driver-drowsiness-dataset (face images only, leakage-free group split)

| Run | Accuracy | Drowsy precision | Drowsy recall | Drowsy F1 | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Run 1: Rule-based landmarks (EAR/MAR/pitch thresholds) | 65.5% | 63.5% | 77.5% | 69.8% | 64.8% | – |
| Run 2: CNN, YOLOv8n-cls (fine-tuned) | 88.5% | 84.6% | 94.8% | 89.4% | 88.4% | 0.940 |
| Run 3: Landmark features + gradient boosting | 87.0% | 84.4% | 91.6% | 87.9% | 86.9% | 0.916 |
| Run 4: Ensemble (CNN + landmark classifier) | 88.6% | 84.5% | 95.3% | 89.6% | 88.5% | 0.945 |
