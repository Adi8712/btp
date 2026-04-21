import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import Dataset
from tqdm import tqdm

from models import SS_UIE, BLUE_Net

IMG_SIZE = (256, 256)
BLUENET_PATH = "./models/bluenet.pth"
SSUIE_PATH = "./models/ss-uie.pth"
CSV_NAME = "results.csv"
DATASETS = {
    "LSUI": {"input": "./dataset/LSUI/input", "gt": "./dataset/LSUI/GT"},
    "UIEB": {"input": "./dataset/UIEB/raw-890", "gt": "./dataset/UIEB/reference-890"},
}

SAVE_OUTPUTS = True
USE_AMP = torch.cuda.is_available()
WARMUP_RUNS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

DEVICE = device


def strip_module(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


class GuidedFilter:
    def __init__(self, I, radius, epsilon):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        if img.dtype == np.float32:
            return img
        return np.float32(img) / 255.0

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir**2, (r, r)) - self._Ir_mean**2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig**2, (r, r)) - self._Ig_mean**2 + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib**2, (r, r)) - self._Ib_mean**2 + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        self._Irr_inv = Irr_inv / cov
        self._Irg_inv = Irg_inv / cov
        self._Irb_inv = Irb_inv / cov
        self._Igg_inv = Igg_inv / cov
        self._Igb_inv = Igb_inv / cov
        self._Ibb_inv = Ibb_inv / cov

    def filter(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p = self._toFloatImg(p)
        p_mean = cv2.blur(p, (r, r))
        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar = cv2.blur(ar, (r, r))
        ag = cv2.blur(ag, (r, r))
        ab = cv2.blur(ab, (r, r))
        b = cv2.blur(b, (r, r))

        return ar * Ir + ag * Ig + ab * Ib + b


def get_attenuation(image, gamma=1.2):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    vals = [np.mean(1 - b**gamma), np.mean(1 - g**gamma), np.mean(1 - r**gamma)]
    return np.argsort(vals)


def getMaxChannel(img, blockSize):
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img, kernel)


def DepthMap(img, blockSize, idx):
    c_star = img[:, :, idx[-1]]
    c = np.maximum(img[:, :, idx[0]], img[:, :, idx[1]])
    max1 = getMaxChannel(c_star, blockSize)
    max2 = getMaxChannel(c, blockSize)
    return max1 - max2


def estimateBackgroundLight(image):
    idx = get_attenuation(image)
    depth = DepthMap(image, 9, idx)
    pos = np.unravel_index(np.argmin(depth), depth.shape)
    return image[pos[0], pos[1], :]


def generator(args):
    fname, input_dir, t_dir, b_dir = args
    stem = Path(fname).stem
    t_path = os.path.join(t_dir, stem + ".png")
    b_path = os.path.join(b_dir, stem + ".png")

    if os.path.exists(t_path) and os.path.exists(b_path):
        return

    img = cv2.imread(os.path.join(input_dir, fname))
    if img is None:
        return

    img = cv2.resize(img, IMG_SIZE)
    image = img / 255.0

    idx = get_attenuation(image)
    depth = DepthMap(image, 9, idx)
    t = depth + (1 - np.max(depth))
    t = np.clip(t, 0.1, 0.9)
    t = GuidedFilter(image * 255, 50, 0.001).filter(t * 255)

    A = estimateBackgroundLight(image)
    B = np.ones_like(image) * A

    cv2.imwrite(t_path, np.uint8(np.clip(t, 0, 255)))
    cv2.imwrite(b_path, np.uint8(B * 255))


def generate_priors(name, info):
    input_dir = info["input"]
    base_dir = str(Path(input_dir).parent)
    t_dir = os.path.join(base_dir, "t_prior")
    b_dir = os.path.join(base_dir, "b_prior")
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))
    jobs = [(f, input_dir, t_dir, b_dir) for f in files]

    with ProcessPoolExecutor() as ex:
        list(
            tqdm(
                ex.map(generator, jobs),
                total=len(jobs),
                desc=f"Priors {name}",
            )
        )


class PairedDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        inp = cv2.imread(os.path.join(self.input_dir, fname))
        gt = cv2.imread(os.path.join(self.gt_dir, fname))
        return fname, inp, gt


def load_bluenet():
    model = BLUE_Net(LayerNo=5)
    ckpt = torch.load(BLUENET_PATH, map_location=device)
    model.load_state_dict(strip_module(ckpt["model_state_dict"]))
    model.eval().to(device)
    return model


def load_ssuie():
    model = SS_UIE()
    state = torch.load(SSUIE_PATH, map_location=device)
    model.load_state_dict(strip_module(state))
    model.eval().to(device)
    return model


def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def tensor_to_bgr(t):
    img = t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def compute_metrics(pred, gt):
    gt = cv2.resize(gt, IMG_SIZE)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)
    ssim = structural_similarity(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
    return psnr, ssim


def benchmark(model, model_name, dname, info):
    dataset = PairedDataset(info["input"], info["gt"])
    input_dir = info["input"]
    base_dir = str(Path(input_dir).parent)
    t_dir = os.path.join(base_dir, "t_prior")
    b_dir = os.path.join(base_dir, "b_prior")

    psnrs, ssims = [], []
    infer_times, total_times = [], []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    warmed = False

    for fname, inp, gt in tqdm(dataset, desc=f"{model_name} {dname}"):
        start_total = time.perf_counter()

        x = preprocess(inp)

        if model_name == "BlueNet":
            stem = Path(fname).stem
            tp = cv2.imread(os.path.join(t_dir, stem + ".png"), 0)
            bp = cv2.imread(os.path.join(b_dir, stem + ".png"))

            tp = cv2.resize(tp, IMG_SIZE)
            bp = cv2.resize(bp, IMG_SIZE)

            tp = (
                torch.from_numpy(tp.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            tp = tp.repeat(1, 3, 1, 1).to(device)

            bp = cv2.cvtColor(bp, cv2.COLOR_BGR2RGB)
            bp = (
                torch.from_numpy(bp.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

        if not warmed:
            with torch.inference_mode():
                for _ in range(WARMUP_RUNS):
                    with torch.autocast("cuda", enabled=USE_AMP):
                        if model_name == "BlueNet":
                            _ = model(x, tp, bp)
                        else:
                            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmed = True

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            i0 = time.perf_counter()

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=USE_AMP):
                if model_name == "BlueNet":
                    out = model(x, tp, bp)[0][-1]
                else:
                    out = model(x)

        if device.type == "cuda":
            ender.record()
            torch.cuda.synchronize()
            infer = starter.elapsed_time(ender) / 1000.0
        else:
            infer = time.perf_counter() - i0

        infer_times.append(infer)

        pred = tensor_to_bgr(out)

        if SAVE_OUTPUTS:
            save_dir = os.path.join("outputs", model_name, dname)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, fname), pred)

        psnr, ssim = compute_metrics(pred, gt)
        psnrs.append(psnr)
        ssims.append(ssim)

        total_times.append(time.perf_counter() - start_total)

    total_params = sum(p.numel() for p in model.parameters())

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    n = len(dataset)

    return {
        "Model": model_name,
        "Dataset": dname,
        "Images": n,
        "PSNR": round(np.mean(psnrs), 4),
        "SSIM": round(np.mean(ssims), 4),
        "Infer Mean (ms)": round(np.mean(infer_times) * 1000, 3),
        "Total Mean (ms)": round(np.mean(total_times) * 1000, 3),
        "FPS": round(n / sum(total_times), 2),
        "Params (M)": round(total_params / 1e6, 3),
        "Peak GPU Mem (MB)": round(peak_mem, 2),
    }


def save_chart(df, filename="metrics.png"):
    metrics = ["FPS", "PSNR", "SSIM"]
    datasets = df["Dataset"].unique()
    models = df["Model"].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            vals = []
            for d in datasets:
                row = df[(df["Dataset"] == d) & (df["Model"] == model)]
                vals.append(row.iloc[0][metric] if not row.empty else 0)

            pos = x - 0.4 + width / 2 + i * width
            bars = ax.bar(pos, vals, width=width, label=model)

            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(metric, fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Value", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for name, info in DATASETS.items():
        generate_priors(name, info)

    models = {
        "BlueNet": load_bluenet(),
        "SS-UIE": load_ssuie(),
    }

    rows = []

    for dname, info in DATASETS.items():
        for model_name, model in models.items():
            rows.append(benchmark(model, model_name, dname, info))

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(CSV_NAME, index=False)

    save_chart(df)


if __name__ == "__main__":
    main()
