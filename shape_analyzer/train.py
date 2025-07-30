import torch
from tqdm.auto import tqdm

# Enable TensorFloat‑32 computation on supported hardware
torch.set_float32_matmul_precision("high")

from .config import cfg
from .data import build_qd_dataset, build_qd_holdout, make_loaders
from .model import ShapeNet


def train():
    build_qd_dataset()
    build_qd_holdout()
    dl_tr, dl_va, _ = make_loaders()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ShapeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = float("inf")
    cfg.CKPT_DIR.mkdir(exist_ok=True)
    save_path = cfg.CKPT_DIR / "model.pt"

    for epoch in tqdm(range(cfg.MAX_EPOCHS), desc="Epochs"):
        model.train()
        for xb, yb, *_ in tqdm(dl_tr, desc=f"Train {epoch+1}/{cfg.MAX_EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, *_ in tqdm(dl_va, desc="Validate", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss /= len(dl_va.dataset)
        sch.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return save_path


if __name__ == "__main__":
    train()
