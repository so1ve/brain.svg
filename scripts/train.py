"""Train a small clipped MLP for SVG filters. NumPy is the only dependency.

Train on UCI Optdigits and MNIST, keeping validation subsets separate from
training and preserving both datasets' original held-out splits.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from mnist import load_mnist

DATA_URL = "https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip"
CENTER = 128 / 255


def quantize(values):
    return np.floor(np.clip(values, 0, 1) * 255 + 0.5) / 255


def load_optdigits():
    cache = Path(".cache/digits")
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / "optdigits.zip"
    if not archive_path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_path.write_bytes(response.read())
    compressed = archive_path.read_bytes()
    result = []
    with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
        for name in ["optdigits.tra", "optdigits.tes"]:
            entry = next(n for n in archive.namelist() if n.split("/")[-1] == name)
            data = np.loadtxt(io.BytesIO(archive.read(entry)), delimiter=",")
            result.append((quantize(data[:, :64] / 16), data[:, 64].astype(int)))
    return result, hashlib.sha256(compressed).hexdigest()


def filter_dense(inputs, weights, bias, hidden):
    norms = np.abs(weights).sum(axis=0) + np.abs(bias)
    scales = 0.4 / np.maximum(1, norms if hidden else norms.max())
    dot = inputs[:, :3] @ weights[:3] + bias
    accumulator = quantize(CENTER + dot * scales)
    for start in range(3, inputs.shape[1], 3):
        dot = inputs[:, start : start + 3] @ weights[start : start + 3]
        term = quantize(CENTER + dot * scales)
        accumulator = quantize(accumulator + term - CENTER)
    return (accumulator - CENTER) / scales


def forward(inputs, params, quantized):
    w1, b1, w2, b2 = params
    z1 = filter_dense(inputs, w1, b1, True) if quantized else inputs @ w1 + b1
    hidden = quantize(z1) if quantized else np.clip(z1, 0, 1)
    logits = filter_dense(hidden, w2, b2, False) if quantized else hidden @ w2 + b2
    return z1, hidden, logits


def metrics(inputs, labels, params, quantized):
    logits = forward(inputs, params, quantized)[2]
    scores = logits - logits.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(scores).sum(axis=1))
    loss = np.mean(log_partition - scores[np.arange(len(labels)), labels])
    return float(np.mean(np.argmax(logits, axis=1) == labels)), float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--quantized-after", type=int, default=30)
    parser.add_argument("--output", default="src/model")
    args = parser.parse_args()
    (training, test), archive_hash = load_optdigits()
    x_all, y_all = training
    x_test, y_test = test
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y_all))
    val_index, train_index = order[:384], order[384:]
    x, y = x_all[train_index], y_all[train_index]
    x_val, y_val = x_all[val_index], y_all[val_index]
    (mnist_training, mnist_test), mnist_hashes = load_mnist()
    mx, my = mnist_training
    mnist_order = rng.permutation(len(my))
    mval, mtrain = mnist_order[:2000], mnist_order[2000:]
    unique_training = len(y) + len(mtrain)
    # Repeat only original training rows to retain the UCI domain while
    # adding diverse MNIST handwriting. Validation and test rows stay out.
    x = np.concatenate([np.tile(x, (3, 1)), mx[mtrain]])
    y = np.concatenate([np.tile(y, 3), my[mtrain]])
    x_val = np.concatenate([x_val, mx[mval]])
    y_val = np.concatenate([y_val, my[mval]])
    params = [
        rng.normal(0, 0.13, (64, args.hidden)),
        np.full(args.hidden, 0.35),
        rng.normal(0, 0.22, (args.hidden, 10)),
        np.zeros(10),
    ]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best_key = (-1, -float("inf"))
    best = None
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        quantized = epoch > args.quantized_after
        rate = 0.003 if not quantized else 0.001
        for batch in np.array_split(
            rng.permutation(len(y)), int(np.ceil(len(y) / 128))
        ):
            xb, yb = x[batch], y[batch]
            z1, hidden, logits = forward(xb, params, quantized)
            scores = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            probabilities[np.arange(len(batch)), yb] -= 1
            probabilities /= len(batch)
            hidden_grad = (probabilities @ params[2].T) * ((z1 > 0) & (z1 < 1))
            grads = [
                xb.T @ hidden_grad + 0.0001 * params[0],
                hidden_grad.sum(axis=0),
                hidden.T @ probabilities + 0.0001 * params[2],
                probabilities.sum(axis=0),
            ]
            step += 1
            for i, (parameter, grad) in enumerate(zip(params, grads)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad**2
                parameter -= rate * (first[i] / (1 - 0.9**step)) / (
                    np.sqrt(second[i] / (1 - 0.999**step)) + 1e-8
                )
        if epoch % 10 == 0 or epoch == args.epochs:
            accuracy, loss = metrics(x_val, y_val, params, True)
            row = {
                "epoch": epoch,
                "quantized": quantized,
                "validationAccuracy": accuracy,
                "validationLoss": loss,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            key = (accuracy, -loss)
            if key > best_key:
                best_key = key
                best = [p.copy() for p in params]
                best_epoch = epoch
    assert best is not None
    layers = []
    for weights, bias, activation in [
        (best[0], best[1], "clip"),
        (best[2], best[3], "scores"),
    ]:
        layers.append({
            "weights": np.round(weights.T, 9).tolist(),
            "bias": np.round(bias, 9).tolist(),
            "activation": activation,
        })
    payload = {"inputSize": 64, "layers": layers}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "seed": args.seed,
        "epochs": args.epochs,
        "bestEpoch": best_epoch,
        "hidden": args.hidden,
        "trainingSamples": len(y),
        "validationSamples": len(y_val),
        "testSamples": len(y_test),
        "validationAccuracy": best_key[0],
        "floatTestAccuracy": metrics(x_test, y_test, best, False)[0],
        "filterReferenceTestAccuracy": metrics(x_test, y_test, best, True)[0],
        "archiveSha256": archive_hash,
        "datasetUrl": DATA_URL,
        "history": history,
        "dataset": "mixed",
        "preprocessing": "moments-deskew-area-v2",
        "uniqueTrainingSamples": unique_training,
        "mnistArchiveSha256": mnist_hashes,
        "mnistTestSamples": len(mnist_test[1]),
        "mnistFloatTestAccuracy": metrics(*mnist_test, best, False)[0],
        "mnistFilterReferenceTestAccuracy": metrics(*mnist_test, best, True)[0],
    }
    model = {"id": digest, **payload, "training": report}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "digits.json").write_text(json.dumps(model, separators=(",", ":")) + "\n")
    report_path = (
        Path("artifacts/training.json")
        if args.output == "src/model"
        else output / "training.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
