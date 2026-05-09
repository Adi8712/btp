import importlib.util
import os
import sys
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


def load_all_models():
    models = {}
    for d in MODELS_ROOT.iterdir():
        if d.is_dir():
            try:
                models[d.name] = load_model_from_folder(d)
                print(f"Loaded {d.name}")
            except Exception as e:
                print(f"Failed {d.name}: {e}")
    return models


def resize_if_needed(img):
    return cv2.resize(img, IMG_SIZE)


def imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return resize_if_needed(img)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def tensor_to_bgr(batch):
    batch = batch.detach().float().cpu().clamp(0, 1)
    batch = batch.permute(0, 2, 3, 1).numpy()
    return [(img * 255).astype(np.uint8) for img in batch]


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
    h, w = channel.shape
    h = h - h % block
    w = w - w % block
    channel = channel[:h, :w]

    k1 = h // block
    k2 = w // block
    score = 0.0
    eps = 1e-8

    for i in range(k1):
        for j in range(k2):
            patch = channel[i * block : (i + 1) * block, j * block : (j + 1) * block]
            mn = patch.min()
            mx = patch.max()
            if mn > 0:
                score += np.log(mx / (mn + eps))

    return 2.0 * score / (k1 * k2)


def logamee(gray, block=8):
    h, w = gray.shape
    h = h - h % block
    w = w - w % block
    gray = gray[:h, :w]

    k1 = h // block
    k2 = w // block
    score = 0.0
    eps = 1e-8

    for i in range(k1):
        for j in range(k2):
            patch = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            mn = patch.min()
            mx = patch.max()
            if mx > mn:
                score += np.log((mx - mn) / (mx + mn + eps) + eps)

    return -score / (k1 * k2)


def uicm(img):
    b, g, r = cv2.split(img.astype(np.float32))

    rg = np.sort((r - g).flatten())
    yb = np.sort(((r + g) / 2 - b).flatten())

    a = 0.1
    t1 = int(a * len(rg))
    t2 = int(a * len(yb))

    rg = rg[t1:-t1]
    yb = yb[t2:-t2]

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    var_rg = np.var(rg)
    var_yb = np.var(yb)

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
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
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
            [f.name for f in self.input_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
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
            target_path = find_file(self.target_dir, stem)
            sample["target"] = preprocess(imread(target_path))

        if self.use_priors:
            t_path = find_file(self.input_dir.parent / "t_prior", stem)
            b_path = find_file(self.input_dir.parent / "b_prior", stem)

            sample["t_prior"] = preprocess(imread(t_path))
            sample["b_prior"] = preprocess(imread(b_path))

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

    if model_name in PRIOR_MODELS:
        t = torch.randn(1, 3, IMG_SIZE[1], IMG_SIZE[0], device=DEVICE)
        b = torch.randn(1, 3, IMG_SIZE[1], IMG_SIZE[0], device=DEVICE)

        for _ in range(WARMUP_RUNS):
            _ = model(x, t, b)
    else:
        for _ in range(WARMUP_RUNS):
            _ = model(x)

    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward_model(name, model, batch):
    x = batch["input"].to(DEVICE, non_blocking=True)
    if DEVICE == "cuda":
        starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
        starter.record()
    if name in PRIOR_MODELS:
        t = batch["t_prior"].to(DEVICE)
        b = batch["b_prior"].to(DEVICE)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        y = model(x, t, b)
    else:
        y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    if isinstance(y, list):
        y = y[-1]
    if DEVICE == "cuda":
        ender.record()
        torch.cuda.synchronize()
        dt = starter.elapsed_time(ender) / 1000.0
    else:
        dt = 0.0
    return y, dt


def benchmark_dataset(model_name, model, ds_path, paired, metrics):
    dataset = BenchmarkDataset(
        ds_path / "input",
        ds_path / "target" if paired else None,
        use_priors=(model_name in PRIOR_MODELS),
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
    psnr_scores, ssim_scores, msssim_scores, lpips_scores = [], [], [], []
    niqe_scores, musiq_scores, topiq_scores, uranker_scores = [], [], [], []
    uciqe_scores, uiqm_scores = [], []
    total_images, total_time = 0, 0.0
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch in tqdm(loader, desc=f"{model_name} | {ds_path.name}"):
        preds, dt = forward_model(model_name, model, batch)
        preds = preds.clamp(0, 1)
        total_time += dt
        total_images += preds.size(0)
        niqe_scores.extend(metrics["niqe"](preds).view(-1).cpu().tolist())
        musiq_scores.extend(metrics["musiq"](preds).view(-1).cpu().tolist())
        topiq_scores.extend(metrics["topiq_nr"](preds).view(-1).cpu().tolist())
        uranker_scores.extend(metrics["uranker"](preds).view(-1).cpu().tolist())
        if paired:
            gt = batch["target"].to(DEVICE)
            psnr_scores.extend(metrics["psnr"](preds, gt).view(-1).cpu().tolist())
            ssim_scores.extend(metrics["ssim"](preds, gt).view(-1).cpu().tolist())
            msssim_scores.extend(metrics["ms_ssim"](preds, gt).view(-1).cpu().tolist())
            lpips_scores.extend(metrics["lpips"](preds, gt).view(-1).cpu().tolist())
        imgs = tensor_to_bgr(preds)
        for i, img in enumerate(imgs):
            uciqe_scores.append(uciqe(img))
            uiqm_scores.append(uiqm(img))
            if SAVE_IMAGES:
                save_dir = OUTPUT_DIR / model_name / ds_path.name
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / batch["name"][i]), img)
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0
    latency = (total_time / total_images) * 1000
    fps = total_images / total_time
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {
        "Model": model_name,
        "Dataset": ds_path.name,
        "PSNR": np.mean(psnr_scores) if paired else np.nan,
        "SSIM": np.mean(ssim_scores) if paired else np.nan,
        "MS-SSIM": np.mean(msssim_scores) if paired else np.nan,
        "LPIPS": np.mean(lpips_scores) if paired else np.nan,
        "NIQE": np.mean(niqe_scores),
        "MUSIQ": np.mean(musiq_scores),
        "TOPIQ-NR": np.mean(topiq_scores),
        "URanker": np.mean(uranker_scores),
        "UCIQE": np.mean(uciqe_scores),
        "UIQM": np.mean(uiqm_scores),
        "Latency(ms)": latency,
        "FPS": fps,
        "Params(M)": params,
        "GPU Mem(MB)": gpu_mem,
    }


def rank_models(df):
    metric_info = {
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
        "FPS": True,
        "Params(M)": False,
    }
    score = pd.Series(0.0, index=df.index)
    for m, h in metric_info.items():
        score += df[m].rank(ascending=not h, method="min")
    df["RankScore"] = score
    df["FinalRank"] = score.rank(method="min")
    return df.sort_values("FinalRank")


def plot_results(df):
    metrics = [
        "PSNR",
        "SSIM",
        "MS-SSIM",
        "LPIPS",
        "NIQE",
        "MUSIQ",
        "TOPIQ-NR",
        "URanker",
        "UCIQE",
        "UIQM",
        "Latency(ms)",
        "Params(M)",
    ]
    higher_better = {
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
    fig, axes = plt.subplots(4, 3, figsize=(22, 18))
    axes = axes.ravel()
    for i, m in enumerate(metrics):
        ax = axes[i]
        vals = df[m].values
        names = df.index.tolist()
        bars = ax.bar(names, vals)
        if higher_better[m]:
            best_idx = np.nanargmax(vals)
        else:
            best_idx = np.nanargmin(vals)
        for j, b in enumerate(bars):
            h = b.get_height()
            if j == best_idx:
                b.set_hatch("//")
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center")
        arrow = "↑" if higher_better[m] else "↓"
        ax.set_title(f"{m} {arrow}")
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
    models = load_all_models()
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
