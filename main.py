import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyiqa
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODELS_ROOT = Path("models")
REF_ROOT = Path("dataset/testset(ref)")
NONREF_ROOT = Path("dataset/testset(non-ref)")
OUTPUT_DIR = Path("outputs")

SAVE_IMAGES = True
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = min(8, os.cpu_count() or 4)
WARMUP_RUNS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

PRIOR_MODELS = {"BLUE-Net", "OurV1"}

METRIC_INFO = {
    "PSNR": True,
    "SSIM": True,
    "MS-SSIM": True,
    "LPIPS": False,
    "NIQE": False,
    "MUSIQ": True,
    "TOPIQ-NR": True,
    "URanker": True,
    "UCIQE": True,
    "UIQM": True,
    "Latency(ms)": False,
    "Params(M)": False,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_model_from_folder(model_dir):
    model_dir = Path(model_dir)
    model_file = model_dir / "model.py"
    weight_file = model_dir / "weights.pth"
    sys.path.insert(0, str(model_dir))
    try:
        module_name = f"model_{model_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model()
        state = torch.load(weight_file, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        return model
    finally:
        sys.path.pop(0)


def load_models():
    models = {}
    for d in sorted(MODELS_ROOT.iterdir()):
        if d.is_dir():
            try:
                models[d.name] = load_model_from_folder(d)
                print(f"Loaded {d.name}")
            except Exception as e:
                print(f"Failed to load {d.name}: {e}")
    return models


@torch.no_grad()
def validate_model(model_name, model):
    x = torch.randn(1, 3, IMG_SIZE[1], IMG_SIZE[0], device=DEVICE)

    if model_name in PRIOR_MODELS:
        y = model(x, torch.randn_like(x), torch.randn_like(x))
    else:
        y = model(x)

    if isinstance(y, tuple):
        y = y[0]
    if isinstance(y, list):
        y = y[-1]

    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Output is {type(y).__name__}, expected Tensor")
    if y.ndim != 4:
        raise ValueError(f"Output has {y.ndim} dims, expected 4 (B,C,H,W)")
    if y.shape[1] != 3:
        raise ValueError(f"Output has {y.shape[1]} channels, expected 3")
    if y.shape[2:] != x.shape[2:]:
        raise ValueError(
            f"Output spatial {tuple(y.shape[2:])} != input {tuple(x.shape[2:])}"
        )


def validate_models(models):
    validated = {}
    failed = []

    print(f"\n{'─' * 60}")
    print(f"Validating {len(models)} models...")
    print(f"{'─' * 60}")

    for name, model in models.items():
        try:
            validate_model(name, model)
            validated[name] = model
            print(f"- {name}")
        except Exception as e:
            failed.append(name)
            print(f"x {name}: {e}")

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"{'─' * 60}")
    print(f"Validated {len(validated)}/{len(models)} models.")
    if failed:
        print(f"Removed: {', '.join(failed)}")
    print(f"{'─' * 60}\n")

    return validated


def imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.resize(img, IMG_SIZE)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def tensor_to_bgr(batch):
    batch = batch.detach().float().cpu().clamp(0, 1)
    batch = batch.permute(0, 2, 3, 1).numpy()

    return [
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for img in batch
    ]


def _blockify(channel, block):
    h, w = channel.shape
    h -= h % block
    w -= w % block
    channel = channel[:h, :w]
    return (
        channel.reshape(h // block, block, w // block, block)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block, block)
    )


def uciqe(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )

    L = lab[:, :, 0] / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    sat = chroma / np.sqrt(chroma**2 + L**2 + 1e-12)
    mu_s = np.mean(sat)
    con_l = np.percentile(L, 99) - np.percentile(L, 1)

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


def eme(channel, block=8):
    blocks = _blockify(channel, block)
    mn = blocks.min(axis=(1, 2))
    mx = blocks.max(axis=(1, 2))
    eps = 1e-8
    valid = mn > 0
    scores = np.where(valid, np.log(mx / (mn + eps)), 0.0)
    return 2.0 * scores.sum() / len(blocks)


def logamee(gray, block=8):
    blocks = _blockify(gray, block)
    mn = blocks.min(axis=(1, 2))
    mx = blocks.max(axis=(1, 2))
    eps = 1e-8
    valid = mx > mn
    scores = np.where(valid, np.log((mx - mn) / (mx + mn + eps) + eps), 0.0)
    return -scores.sum() / len(blocks)


def uicm(img):
    b, g, r = cv2.split(img.astype(np.float32))

    rg = np.sort((r - g).flatten())
    yb = np.sort(((r + g) / 2 - b).flatten())

    t1 = int(0.1 * len(rg))
    t2 = int(0.1 * len(yb))

    rg = rg[t1:-t1]
    yb = yb[t2:-t2]

    mu_rg, mu_yb = np.mean(rg), np.mean(yb)
    var_rg, var_yb = np.var(rg), np.var(yb)

    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(var_rg + var_yb)


def uiqm(img):
    img = img.astype(np.float32)
    uicm_val = uicm(img)

    r, g, b = cv2.split(img)
    sr = np.abs(cv2.Sobel(r, cv2.CV_32F, 1, 1))
    sg = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 1))
    sb = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 1))
    uism_val = (eme(sr) + eme(sg) + eme(sb)) / 3.0

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    uiconm_val = logamee(gray)

    return float(0.0282 * uicm_val + 0.2953 * uism_val + 3.5753 * uiconm_val)


def find_file(folder, stem):
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No file found for {stem} in {folder}")


class BenchmarkDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, use_priors=False):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir) if target_dir else None
        self.use_priors = use_priors
        self.files = sorted(
            f.name for f in self.input_dir.iterdir() if f.suffix.lower() in IMG_EXTS
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        stem = Path(name).stem

        sample = {
            "name": name,
            "input": preprocess(imread(self.input_dir / name)),
        }

        if self.target_dir:
            sample["target"] = preprocess(imread(find_file(self.target_dir, stem)))

        if self.use_priors:
            sample["t_prior"] = preprocess(
                imread(find_file(self.input_dir.parent / "t_prior", stem))
            )
            sample["b_prior"] = preprocess(
                imread(find_file(self.input_dir.parent / "b_prior", stem))
            )

        return sample


def collate_fn(batch):
    out = {
        "name": [b["name"] for b in batch],
        "input": torch.stack([b["input"] for b in batch]),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])
    if "t_prior" in batch[0]:
        out["t_prior"] = torch.stack([b["t_prior"] for b in batch])
        out["b_prior"] = torch.stack([b["b_prior"] for b in batch])
    return out


@torch.no_grad()
def warmup(model_name, model):
    x = torch.randn(1, 3, IMG_SIZE[1], IMG_SIZE[0], device=DEVICE)
    args = (
        (x, torch.randn_like(x), torch.randn_like(x))
        if model_name in PRIOR_MODELS
        else (x,)
    )
    for _ in range(WARMUP_RUNS):
        model(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward_model(name, model, batch):
    x = batch["input"].to(DEVICE, non_blocking=True)

    use_cuda = DEVICE == "cuda"
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

    if name in PRIOR_MODELS:
        t = batch["t_prior"].to(DEVICE, non_blocking=True)
        b = batch["b_prior"].to(DEVICE, non_blocking=True)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        y = model(x, t, b)
    else:
        y = model(x)

    if isinstance(y, tuple):
        y = y[0]
    if isinstance(y, list):
        y = y[-1]

    if use_cuda:
        ender.record()
        torch.cuda.synchronize()
        dt = starter.elapsed_time(ender) / 1000.0
    else:
        dt = 0.0

    return y, dt


def _count_images(folder):
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)


def has_cached_outputs(model_name, ds_path):
    cache_dir = OUTPUT_DIR / model_name / ds_path.name
    if not cache_dir.is_dir():
        return False
    input_count = _count_images(ds_path / "input")
    cached_count = _count_images(cache_dir)
    return cached_count >= input_count > 0


def compute_quality_metrics(preds, batch, paired, metrics, scores):
    preds = preds.clamp(0, 1)

    scores["NIQE"].extend(metrics["niqe"](preds).view(-1).cpu().tolist())
    scores["MUSIQ"].extend(metrics["musiq"](preds).view(-1).cpu().tolist())
    scores["TOPIQ-NR"].extend(metrics["topiq_nr"](preds).view(-1).cpu().tolist())
    scores["URanker"].extend(metrics["uranker"](preds).view(-1).cpu().tolist())

    if paired:
        gt = batch["target"].to(DEVICE, non_blocking=True)
        scores["PSNR"].extend(metrics["psnr"](preds, gt).view(-1).cpu().tolist())
        scores["SSIM"].extend(metrics["ssim"](preds, gt).view(-1).cpu().tolist())
        scores["MS-SSIM"].extend(metrics["ms_ssim"](preds, gt).view(-1).cpu().tolist())
        scores["LPIPS"].extend(metrics["lpips"](preds, gt).view(-1).cpu().tolist())

    for img in tensor_to_bgr(preds):
        scores["UCIQE"].append(uciqe(img))
        scores["UIQM"].append(uiqm(img))


def benchmark_dataset(model_name, model, ds_path, paired, metrics):
    use_priors = model_name in PRIOR_MODELS
    cached = SAVE_IMAGES and has_cached_outputs(model_name, ds_path)

    dataset = BenchmarkDataset(
        ds_path / "input",
        ds_path / "target" if paired else None,
        use_priors=(use_priors and not cached),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate_fn,
    )

    scores = defaultdict(list)
    total_images, total_time = 0, 0.0
    cache_dir = OUTPUT_DIR / model_name / ds_path.name
    label = f"{model_name} | {ds_path.name}"

    if DEVICE == "cuda" and not cached:
        torch.cuda.reset_peak_memory_stats()

    if cached:
        for batch in tqdm(loader, desc=f"{label} (cached)"):
            names = batch["name"]
            pred_tensors = []
            for name in names:
                stem = Path(name).stem
                out_path = find_file(cache_dir, stem)
                pred_tensors.append(preprocess(imread(str(out_path))))
            preds = torch.stack(pred_tensors).to(DEVICE, non_blocking=True)
            total_images += preds.size(0)
            compute_quality_metrics(preds, batch, paired, metrics, scores)
    else:
        if SAVE_IMAGES:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for batch in tqdm(loader, desc=label):
            preds, dt = forward_model(model_name, model, batch)
            preds = preds.clamp(0, 1)
            total_time += dt
            total_images += preds.size(0)
            compute_quality_metrics(preds, batch, paired, metrics, scores)

            if SAVE_IMAGES:
                for i, img in enumerate(tensor_to_bgr(preds)):
                    cv2.imwrite(str(cache_dir / batch["name"][i]), img)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    gpu_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if DEVICE == "cuda" and not cached
        else np.nan
    )

    if cached or total_time == 0:
        latency = np.nan
    else:
        latency = (total_time / total_images) * 1000

    return {
        "Model": model_name,
        "Dataset": ds_path.name,
        "PSNR": np.mean(scores["PSNR"]) if paired else np.nan,
        "SSIM": np.mean(scores["SSIM"]) if paired else np.nan,
        "MS-SSIM": np.mean(scores["MS-SSIM"]) if paired else np.nan,
        "LPIPS": np.mean(scores["LPIPS"]) if paired else np.nan,
        "NIQE": np.mean(scores["NIQE"]),
        "MUSIQ": np.mean(scores["MUSIQ"]),
        "TOPIQ-NR": np.mean(scores["TOPIQ-NR"]),
        "URanker": np.mean(scores["URanker"]),
        "UCIQE": np.mean(scores["UCIQE"]),
        "UIQM": np.mean(scores["UIQM"]),
        "Latency(ms)": latency,
        "Params(M)": params,
        "GPU Mem(MB)": gpu_mem,
    }


def rank_models(df):
    score = pd.Series(0.0, index=df.index)
    for m, higher in METRIC_INFO.items():
        if m in df.columns:
            score += df[m].rank(ascending=not higher, method="min")
    df["RankScore"] = score
    df["FinalRank"] = score.rank(method="min")
    return df.sort_values("FinalRank")


def plot_results(df):
    plot_metrics = [m for m in METRIC_INFO if m in df.columns and m != "GPU Mem(MB)"]
    ncols = 3
    nrows = -(-len(plot_metrics) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5 * nrows))
    axes = axes.ravel()

    for i, m in enumerate(plot_metrics):
        ax = axes[i]
        vals = df[m].values
        names = df.index.tolist()
        bars = ax.bar(names, vals)

        higher = METRIC_INFO[m]
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) > 0:
            best_idx = int(np.nanargmax(vals) if higher else np.nanargmin(vals))
            for j, b in enumerate(bars):
                h = b.get_height()
                if j == best_idx:
                    b.set_hatch("//")
                if np.isfinite(h):
                    ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center")

        arrow = "↑" if higher else "↓"
        ax.set_title(f"{m} {arrow}")

    for j in range(len(plot_metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=300)
    plt.show()


def main():
    metrics = {
        "psnr": pyiqa.create_metric("psnr", device=DEVICE),
        "ssim": pyiqa.create_metric("ssim", device=DEVICE),
        "ms_ssim": pyiqa.create_metric("ms_ssim", device=DEVICE),
        "lpips": pyiqa.create_metric("lpips", device=DEVICE),
        "niqe": pyiqa.create_metric("niqe", device=DEVICE),
        "musiq": pyiqa.create_metric("musiq", device=DEVICE),
        "topiq_nr": pyiqa.create_metric("topiq_nr", device=DEVICE),
        "uranker": pyiqa.create_metric("uranker", device=DEVICE),
    }

    print("Loading models")
    models = load_models()
    if not models:
        print("No models loaded")
        return

    models = validate_models(models)
    if not models:
        print("No models passed validation")
        return

    all_results = []
    for name, model in models.items():
        warmup(name, model)
        for ds in sorted(REF_ROOT.iterdir()):
            if ds.is_dir():
                all_results.append(benchmark_dataset(name, model, ds, True, metrics))
        for ds in sorted(NONREF_ROOT.iterdir()):
            if ds.is_dir():
                all_results.append(benchmark_dataset(name, model, ds, False, metrics))

    df = pd.DataFrame(all_results)
    df.to_csv("metrics_per_dataset.csv", index=False)
    overall_df = df.groupby("Model").mean(numeric_only=True)
    overall_df = rank_models(overall_df)
    overall_df.to_csv("metrics.csv")
    print(df)
    print("\nOVERALL:\n", overall_df)
    plot_results(overall_df)


if __name__ == "__main__":
    main()
