from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import wandb  # type: ignore
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

from radiology_cls import (
    NUM_CLASSES,
    LABEL_NAMES,
    ChestXrayDataset,
    load_train_df,
    train_val_split,
    ensure_dir,
    BaseModel,
)
from radiology_cls.preprocessing import encode_labels, train_transform, val_transform, masked_bce_loss
from radiology_cls.eval import compute_classification_metrics, plot_auroc, plot_pr_curve, save_metrics


class VanillaCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, dropout_prob=0.2):
        super().__init__()
        
        # incoming images are (batch_size, in_channels, height, width)
        # dimension comments assume incoming images are 256 x 256 and 
        # just illustrate the dimension of the feature map, not num channels
        self.features = nn.Sequential(
            ## block 1
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), padding=1), # 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1), # 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2), # 128 x 128
            nn.Dropout(dropout_prob),

            ## block 2
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1), # 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), # 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2), # 64 x 64
            nn.Dropout(dropout_prob),

            ## block 3
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1), # 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1), # 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2), # 32 x 32
            nn.Dropout(dropout_prob),

            ## block 4
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1), # 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1), # 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(2), # 16 x 16
            nn.Dropout(dropout_prob),

            nn.AdaptiveAvgPool2d(1),  # (256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 256, 1, 1)
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),  # (B, 9)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VanillaCNNModel(BaseModel):

    @property
    def name(self):
        return "vanilla-cnn"

    def __init__(
        self,
        image_size=256,
        batch_size=32,
        lr=1e-4,
        epochs=50,
        dropout_prob=0.2,
        val_frac=0.1,
        seed=42,
        num_workers=4,
        uncertain_as=1.0,
        **kwargs,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout_prob = dropout_prob
        self.val_frac = val_frac
        self.seed = seed
        self.num_workers = num_workers
        self.uncertain_as = uncertain_as

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VanillaCNN(
            in_channels=1, dropout_prob=self.dropout_prob,
        ).to(self.device)

    def train(self, run_dir: Path) -> dict:
        use_amp = self.device.type == "cuda"
        print(f"Device: {self.device}")
        if use_amp:
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print("Mixed precision: float16 (AMP enabled)")
            torch.backends.cudnn.benchmark = True

        ## prepare the dataset
        df = load_train_df()
        train_df, val_df = train_val_split(df, self.val_frac, self.seed)

        y_train = encode_labels(train_df, list(LABEL_NAMES), uncertain_as=self.uncertain_as)
        y_val = encode_labels(val_df, list(LABEL_NAMES), uncertain_as=float("nan"))

        train_ds = ChestXrayDataset(
            train_df, y_train, train_transform(self.image_size),
            image_size=self.image_size,
        )
        val_ds = ChestXrayDataset(
            val_df, y_val, val_transform(self.image_size),
            image_size=self.image_size,
        )

        use_persistent = self.num_workers > 0
        loader_kwargs = dict(
            num_workers=self.num_workers,
            pin_memory=use_amp,
            persistent_workers=use_persistent,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, **loader_kwargs,
        )

        ## set up optimizer and train!
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        final_val_loss = float("nan")
        best_val_logits_all: list[torch.Tensor] | None = None
        best_val_labels_all: list[torch.Tensor] | None = None

        with tqdm(range(self.epochs), desc="epochs", unit="epoch") as epoch_bar:
            for epoch in epoch_bar:
                epoch_num = epoch + 1
                self.model.train()
                train_loss_sum = 0.0
                train_batches = 0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits = self.model(x)
                        loss = criterion(logits, y)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item()
                    train_batches += 1

                train_loss = train_loss_sum / train_batches

                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                val_logits_all: list[torch.Tensor] = []
                val_labels_all: list[torch.Tensor] = []

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            logits = self.model(x)
                        loss = masked_bce_loss(logits.float(), y)

                        val_loss_sum += loss.item()
                        val_batches += 1
                        val_logits_all.append(logits.float().cpu())
                        val_labels_all.append(y.cpu())

                val_loss = val_loss_sum / val_batches
                final_val_loss = val_loss

                epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

                if val_loss < best_val_loss:
                    previous_best = best_val_loss
                    best_val_loss = val_loss
                    best_epoch = epoch_num
                    best_val_logits_all = [batch.clone() for batch in val_logits_all]
                    best_val_labels_all = [batch.clone() for batch in val_labels_all]
                    torch.save(self.model.state_dict(), run_dir / "best.pt")
                    previous_msg = "n/a" if previous_best == float("inf") else f"{previous_best:.4f}"
                    log.info(
                        "Saved new best checkpoint at epoch %d: val_loss=%.4f (previous best=%s)",
                        best_epoch,
                        best_val_loss,
                        previous_msg,
                    )

                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "epoch": epoch_num,
                })

        ## generate evaluation plots
        torch.save(self.model.state_dict(), run_dir / "last.pt")
        if best_val_logits_all is None or best_val_labels_all is None:
            raise RuntimeError("No best checkpoint was saved; epochs must be at least 1")

        state_dict = torch.load(run_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)

        y_true = torch.cat(best_val_labels_all).numpy()
        y_pred = torch.sigmoid(torch.cat(best_val_logits_all)).numpy()

        plots_dir = ensure_dir(run_dir / "plots")
        metrics = compute_classification_metrics(y_true, y_pred, list(LABEL_NAMES))
        save_metrics(metrics, run_dir)

        auroc_fig = plot_auroc(y_true, y_pred, list(LABEL_NAMES), plots_dir)
        pr_fig = plot_pr_curve(y_true, y_pred, list(LABEL_NAMES), plots_dir)
        import matplotlib.pyplot as plt # type: ignore
        plt.close(auroc_fig)
        plt.close(pr_fig)

        log.info(f"Macro AUROC: {metrics.get('macro_auroc', 'n/a')}")
        log.info(f"Macro PR-AUC: {metrics.get('macro_pr_auc', 'n/a')}")
        log.info(
            "Best checkpoint: epoch %d val_loss=%.4f; final epoch val_loss=%.4f",
            best_epoch,
            best_val_loss,
            final_val_loss,
        )

        return {
            "metric_checkpoint": "best.pt",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_epoch": self.epochs,
            "final_val_loss": final_val_loss,
            "macro_auroc": metrics.get("macro_auroc"),
            "macro_pr_auc": metrics.get("macro_pr_auc"),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        use_amp = self.device.type == "cuda"
        ds = ChestXrayDataset(
            df, labels=None, transform=val_transform(self.image_size),
            image_size=self.image_size,
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=use_amp,
            persistent_workers=self.num_workers > 0,
        )

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.model(x)
                all_preds.append(torch.sigmoid(logits.float()).cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    @classmethod
    def from_checkpoint(cls, run_dir: Path, **kwargs) -> "VanillaCNNModel":
        import yaml

        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        config.pop("model_class", None)
        config.pop("run_name", None)
        wrapper = cls(**config)

        state_dict = torch.load(run_dir / "best.pt", map_location=wrapper.device)
        wrapper.model.load_state_dict(state_dict)

        return wrapper
