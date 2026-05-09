from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_LIST = [
    "test-RUIE-unpaired/ruie_A_342.jpg",
    "test-EUVP-unpaired/unpaired_euvp_test_9012up.jpg",
    "test-EUVP-unpaired/unpaired_euvp_test_9023up.jpg",
    "test-EUVP-unpaired/unpaired_euvp_test_9044up.jpg",
    "test-UIEB-unpaired/5.png",
    "test-UIEB-unpaired/432.png",
    "test-UIEB-unpaired/3051.png",
    "test-UIEB-unpaired/13259.png",
    "test-UIEB-unpaired/15774.png",
]

MODELS = [
    "BLUE-Net",
    "PUIE-Net",
    "Shallow-UWnet",
    "SS-UIE",
    "U-shape",
    "USUIR",
    "OurV1",
    "OurV2",
]

DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("outputs")

COLUMNS = ["Input", *MODELS]


def dataset_type(name):
    return "testset(non-ref)" if "unpaired" in name else "testset(ref)"


def load_image(path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        return np.full((size, size, 3), 240, dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


rows = len(IMAGE_LIST)
cols = len(COLUMNS)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(cols * 2.4, rows * 2.4),
    squeeze=False,
)

for ax, title in zip(axes[0], COLUMNS):
    ax.set_title(title, fontsize=14, fontweight="bold")

for r, item in enumerate(IMAGE_LIST):
    dataset_name, filename = item.split("/", 1)

    paths = [
        DATASET_ROOT / dataset_type(dataset_name) / dataset_name / "input" / filename,
        *[OUTPUT_ROOT / model / dataset_name / filename for model in MODELS],
    ]

    for ax, path in zip(axes[r], paths):
        img = load_image(path)

        ax.imshow(img)
        ax.axis("off")

        if not path.exists():
            ax.text(
                0.5,
                0.5,
                "Missing",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="red", alpha=0.85),
            )

        ax.set_xticks([])
        ax.set_yticks([])

fig.subplots_adjust(wspace=0.02, hspace=0.05)

plt.savefig(
    "comparison.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
