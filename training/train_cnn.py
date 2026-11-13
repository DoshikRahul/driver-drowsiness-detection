"""
Run 2: train a deep-learning drowsiness classifier (YOLOv8n-cls, transfer learning).

Input : datasets/driver_drowsiness/{train,valid,test}/{drowsy,not_drowsy}
Output: runs/classify/drowsiness_yolov8n_cls/   (weights, curves, confusion matrices)
        runs/classify/drowsiness_yolov8n_cls/test_metrics.json
        ml_model/models/drowsiness_cls.onnx     (used by the web app via OpenCV DNN)

Usage:  python training/train_cnn.py [--epochs 20] [--imgsz 224] [--batch 64] [--device cpu|0]
        (use --device 0 --batch 256 --cache on a CUDA GPU)
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "driver_drowsiness"
PROJECT = ROOT / "runs" / "classify"
NAME = "drowsiness_yolov8n_cls"
ONNX_OUT = ROOT / "ml_model" / "models" / "drowsiness_cls.onnx"
BASE_WEIGHTS = ROOT / "training" / "yolov8n-cls.pt"  # ImageNet-pretrained, auto-downloaded


def evaluate_test(model, imgsz, device="cpu"):
    """Per-image predictions on the held-out test split -> metrics dict."""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    names = model.names  # {0: 'drowsy', 1: 'not_drowsy'}
    y_true, y_prob, times = [], [], []
    for idx, cls in names.items():
        files = sorted((DATA / "test" / cls).glob("*.jpg"))
        for i in range(0, len(files), 64):
            batch = [str(f) for f in files[i:i + 64]]
            t0 = time.perf_counter()
            res = model.predict(batch, imgsz=imgsz, verbose=False, device=device)
            times.append((time.perf_counter() - t0) / len(batch))
            for r in res:
                y_prob.append(float(r.probs.data[0]))  # P(drowsy)
                y_true.append(idx)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.where(y_prob >= 0.5, 0, 1)
    labels = [names[0], names[1]]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, digits=4)
    return {
        "split": "test",
        "n_images": int(len(y_true)),
        "accuracy": round(report["accuracy"], 4),
        "drowsy_precision": round(report["drowsy"]["precision"], 4),
        "drowsy_recall": round(report["drowsy"]["recall"], 4),
        "drowsy_f1": round(report["drowsy"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "roc_auc": round(float(roc_auc_score(y_true == 0, y_prob)), 4),
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": confusion_matrix(y_true, y_pred).tolist()},
        "ms_per_image_batched": round(1000 * float(np.mean(times)), 2),
        "device": str(device),
        "report": report,
    }


def main(args):
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    model = YOLO(str(BASE_WEIGHTS))  # ImageNet-pretrained, downloaded on first use
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache="ram" if args.cache else False,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        patience=5,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        seed=0,
        deterministic=False,
        plots=True,
        fliplr=0.5,
        hsv_v=0.3,
        erasing=0.2,
        verbose=True,
    )
    run_dir = PROJECT / NAME
    best = YOLO(str(run_dir / "weights" / "best.pt"))

    metrics = evaluate_test(best, args.imgsz, args.device)
    metrics["model"] = "YOLOv8n-cls (ImageNet-pretrained, fine-tuned)"
    metrics["imgsz"] = args.imgsz
    metrics["epochs_requested"] = args.epochs
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))

    onnx_path = best.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, ONNX_OUT)
    (ONNX_OUT.with_suffix(".json")).write_text(json.dumps(
        {"names": best.names, "imgsz": args.imgsz, "source_run": str(run_dir.relative_to(ROOT)),
         "test_accuracy": metrics["accuracy"], "test_roc_auc": metrics["roc_auc"]}, indent=2))
    print(f"ONNX model copied to {ONNX_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--device", default="cpu", help="'cpu' or a CUDA device index such as 0")
    ap.add_argument("--cache", action="store_true", help="cache decoded images in RAM (faster, ~2 GB)")
    main(ap.parse_args())
