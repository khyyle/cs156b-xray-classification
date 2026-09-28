# %% [markdown]
# # Grad-CAM: DenseNet121 512 Regression Model
# 
# This notebook inspects where the best current DenseNet121 model is looking for each pathology. It loads the run from `runs/densenet121-pretrained_20260427_013323`, reconstructs train val split, and then overlays Grad-CAM (Selvaraju et al. 2016) heatmaps on selected validation X-rays.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from radiology_cls import LABEL_NAMES, load_image, load_train_df, train_val_split
from radiology_cls.preprocessing import encode_regression_labels, val_transform
from radiology_cls.utils import PROJECT_ROOT
from radiology_cls.models.torchvision_cnn import TorchvisionCNNModel

RUN_DIR = PROJECT_ROOT / "runs" / "densenet121-pretrained_20260427_013323"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project root: {PROJECT_ROOT}")
print(f"Run dir: {RUN_DIR}")
print(f"Device: {DEVICE}")

# %% [markdown]
# ## Load Model And Validation Split

# %%
assert RUN_DIR.exists(), f"Run directory does not exist: {RUN_DIR}"
assert (RUN_DIR / "config.yaml").exists(), "Missing config.yaml"
assert (RUN_DIR / "best.pt").exists(), "Missing best.pt"

with open(RUN_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

model = TorchvisionCNNModel.from_checkpoint(RUN_DIR)
model.model.to(DEVICE)
model.model.eval()

print(config)

# %%
df = load_train_df()
train_df, val_df = train_val_split(df, val_frac=model.val_frac, seed=model.seed)

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
assert train_patients.isdisjoint(val_patients), "Patient leakage between train and validation split"
assert len(val_df) > 0, "Validation split is empty"

labels = encode_regression_labels(val_df, list(LABEL_NAMES))
assert labels.shape == (len(val_df), len(LABEL_NAMES))

print(f"Validation rows: {len(val_df):,}")
print(f"Validation patients: {val_df['patient_id'].nunique():,}")
print(f"Labels shape: {labels.shape}")

# %%
sample_img = load_image(val_df.iloc[0]["image_path"], rgb=model.rgb)
sample_x = val_transform(model.image_size, rgb=model.rgb)(sample_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    sample_logits = model.model(sample_x)
    sample_pred = torch.tanh(sample_logits)

assert sample_logits.shape == (1, len(LABEL_NAMES)), sample_logits.shape
assert torch.isfinite(sample_logits).all(), "Model produced non-finite logits"
assert sample_pred.min().item() >= -1.001 and sample_pred.max().item() <= 1.001, "Tanh predictions outside [-1, 1]"

print(pd.DataFrame({
    "pathology": LABEL_NAMES,
    "raw_logit": sample_logits.squeeze().detach().cpu().numpy(),
    "tanh_prediction": sample_pred.squeeze().detach().cpu().numpy(),
}))

# %% [markdown]
# ## Grad-CAM Helpers
# 
# For DenseNet121, `features.denseblock4` is a good target layer because it is late enough to represent high-level pathology evidence while still retaining spatial structure. The target callable below selects a raw pathology logit before `tanh`.

# %%
class RegressionOutputTarget:
    def __init__(self, target_idx: int) -> None:
        self.target_idx = target_idx

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.target_idx]
        return model_output[:, self.target_idx].sum()


target_layers = [model.model.features.denseblock4]
grad_cam = GradCAM(model=model.model, target_layers=target_layers)

sample_target_idx = LABEL_NAMES.index("Cardiomegaly")
cam = grad_cam(input_tensor=sample_x, targets=[RegressionOutputTarget(sample_target_idx)])[0]

with torch.no_grad():
    logits = model.model(sample_x)

assert cam.shape == (model.image_size, model.image_size), cam.shape
assert np.isfinite(cam).all(), "Grad-CAM produced non-finite values"
assert 0.0 <= cam.min() <= cam.max() <= 1.0 + 1e-6, (cam.min(), cam.max())

# %%
def prepare_image(row: pd.Series) -> tuple[torch.Tensor, np.ndarray]:
    pil_img = load_image(row["image_path"], rgb=model.rgb)
    x = val_transform(model.image_size, rgb=model.rgb)(pil_img).unsqueeze(0).to(DEVICE)
    display_img = np.asarray(pil_img.resize((model.image_size, model.image_size), Image.BILINEAR))
    if display_img.ndim == 2:
        display_img = np.stack([display_img] * 3, axis=-1)
    display_img = display_img.astype(np.float32) / 255.0
    return x, display_img


def find_example(label_name: str, value: float = 1.0, frontal_only: bool = True, seed: int = 42) -> pd.Series:
    subset = val_df[val_df[label_name] == value]
    if frontal_only and "Frontal/Lateral" in subset.columns:
        frontal = subset[subset["Frontal/Lateral"] == "Frontal"]
        if len(frontal) > 0:
            subset = frontal
    if len(subset) == 0:
        raise ValueError(f"No validation example found for {label_name} == {value}")
    return subset.sample(n=1, random_state=seed).iloc[0]


def plot_grad_cam(row: pd.Series, label_name: str, abs_error: float, alpha: float = 0.4) -> None:
    label_idx = list(LABEL_NAMES).index(label_name)
    x, display_img = prepare_image(row)
    cam = grad_cam(input_tensor=x, targets=[RegressionOutputTarget(label_idx)])[0]

    with torch.no_grad():
        logits = model.model(x)
        preds = torch.tanh(logits).squeeze().cpu().numpy()

    target_value = row[label_name]
    overlay = show_cam_on_image(display_img, cam, use_rgb=True, image_weight=1 - alpha)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Input X-ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="magma")
    axes[1].set_title(f"Grad-CAM: {label_name}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = (
        f"{Path(row['Path']).name} | target={target_value} | "
        f"pred={preds[label_idx]:.3f} | abs error={abs_error:.3f} | view={row.get('Frontal/Lateral', 'n/a')}"
    )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_dir = PROJECT_ROOT / "scratch" / "kyle" / "grad_cam_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_label = label_name.replace(" ", "_").replace("/", "_")
    save_path = out_dir / f"plot_{safe_label}_row{row.name}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# %% [markdown]
# # Inspect some random samples

# %%
labels_to_review = [
    "Cardiomegaly",
    "Lung Opacity",
    "Pleural Effusion",
    "Support Devices",
    "Fracture",
]

#for i, label_name in tqdm(enumerate(labels_to_review), desc='label'):
    #row = find_example(label_name, value=1.0, seed=42 + i)
    #plot_grad_cam(row, label_name)

# %% [markdown]
# # Inspect high error samples

# %%
def score_validation_subset(label_name: str, n_subset: int = 512, seed: int = 7) -> pd.DataFrame:
    label_idx = list(LABEL_NAMES).index(label_name)
    subset = val_df.sample(n=min(n_subset, len(val_df)), random_state=seed).reset_index(drop=True)
    rows = []

    with torch.no_grad():
        for _, row in tqdm(subset.iterrows(), desc='row'):
            target = row[label_name]
            if pd.isna(target):
                continue
            x, _ = prepare_image(row)
            logits = model.model(x)
            pred = torch.tanh(logits)[0, label_idx].item()
            rows.append({
                "Path": row["Path"],
                "image_path": row["image_path"],
                "Frontal/Lateral": row.get("Frontal/Lateral", None),
                "target": float(target),
                "pred": pred,
                "abs_error": abs(pred - float(target)),
            })

    scored = pd.DataFrame(rows).sort_values("abs_error", ascending=False).reset_index(drop=True)
    return scored


N_SUBSET = 1024

# %%
TOP_K = 5
for label in LABEL_NAMES:
    failures = score_validation_subset(label, n_subset=N_SUBSET)
    for _, failure in tqdm(failures.head(TOP_K).iterrows(), desc='image'):
        row = val_df[val_df["Path"] == failure["Path"]].iloc[0]
        plot_grad_cam(row, label, failure["abs_error"])


