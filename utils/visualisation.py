import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_and_log_crops(
    crops,                       # list/tuple of tensors: [B,3,H,W], first 2 are global, rest are local
    n_samples=10,
    upsample_locals_to=224,       # set None to keep native local sizes (grid can look uneven)
    denorm_mean=(0.485, 0.456, 0.406),
    denorm_std=(0.229, 0.224, 0.225),
    wandb_run=None,               # e.g. wandb_logger.experiment or wandb module
    wandb_key="debug/crops_grid",
    step=None,
    max_cols=None,                # optionally cap number of columns to log (useful if many locals)
    title_prefix="",              # optional title prefix
):
    assert isinstance(crops, (list, tuple)), "crops must be a list/tuple"
    assert len(crops) >= 2, f"Expected at least 2 crops (globals), got {len(crops)}"

    num_views = len(crops)
    num_globals = 2
    num_locals = max(0, num_views - num_globals)

    B = crops[0].shape[0]
    n = min(n_samples, B)

    # Optionally limit number of columns (e.g., if you have tons of local crops)
    if max_cols is not None:
        num_views = min(num_views, max_cols)
        num_locals = max(0, num_views - num_globals)

    # Prepare visualization tensors (optionally upsample locals for prettier grid)
    viz = []
    for i in range(num_views):
        t = crops[i]
        x = t[:n].detach()

        if i >= num_globals and upsample_locals_to is not None:
            x = F.interpolate(
                x,
                size=(upsample_locals_to, upsample_locals_to),
                mode="bilinear",
                align_corners=False,
            )
        viz.append(x)

    # Build figure grid
    fig_w = max(2.0 * num_views, 6.0)
    fig_h = max(2.0 * n, 3.0)
    fig, axes = plt.subplots(nrows=n, ncols=num_views, figsize=(fig_w, fig_h))

    # Normalize axes indexing
    if n == 1 and num_views == 1:
        axes = [[axes]]
    elif n == 1:
        axes = axes.reshape(1, -1)
    elif num_views == 1:
        axes = axes.reshape(-1, 1)

    # Column titles
    col_titles = ["G1", "G2"] + [f"L{i}" for i in range(1, num_locals + 1)]
    col_titles = col_titles[:num_views]
    for c in range(num_views):
        axes[0, c].set_title(f"{title_prefix}{col_titles[c]}", fontsize=10)

    # Plot images
    for r in range(n):
        for c in range(num_views):
            img = viz[c][r]  # [3,H,W]
            img = _denorm_img(img, denorm_mean, denorm_std)
            img = img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # HWC
            axes[r, c].imshow(img)
            axes[r, c].axis("off")

    plt.tight_layout()

    plt.show()
    return fig

def _denorm_img(x, mean, std):
    mean = torch.tensor(mean, device=x.device).view(3, 1, 1)
    std = torch.tensor(std, device=x.device).view(3, 1, 1)
    return x * std + mean