import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_crops_from_dataset(
    dataset,
    n=10,
    idxs=None,
    *,
    upsample_locals_to=224,
    denorm_mean=(0.485, 0.456, 0.406),
    denorm_std=(0.229, 0.224, 0.225),
    max_cols=None,
    title_prefix="",
):
    """
    Plots a grid of crops for multiple dataset items.

    Expects dataset[i][0] to be a dict with:
      - 'global_crops': list/tuple of tensors [3,H,W] (len>=2)
      - 'local_crops' : list/tuple of tensors [3,h,w] (can be empty)

    One row per sample, columns = 2 globals + locals.
    """

    def _denorm_img(x, mean, std):
        mean = torch.tensor(mean, device=x.device).view(3, 1, 1)
        std = torch.tensor(std, device=x.device).view(3, 1, 1)
        return x * std + mean

    if idxs is None:
        idxs = list(range(n))
    else:
        n = len(idxs)

    items = [dataset[i][0] for i in idxs]

    assert "global_crops" in items[0], "Expected key 'global_crops' in dataset[i][0]"
    assert "local_crops" in items[0], "Expected key 'local_crops' in dataset[i][0]"
    assert len(items[0]["global_crops"]) >= 2, "Expected at least 2 global crops"

    # Assume fixed number of views across items
    g_views = len(items[0]["global_crops"])
    l_views = len(items[0]["local_crops"])

    # Stack into [B,3,H,W] per view
    global_crops = [
        torch.stack([it["global_crops"][v] for it in items], dim=0) for v in range(g_views)
    ]
    local_crops = [
        torch.stack([it["local_crops"][v] for it in items], dim=0) for v in range(l_views)
    ]

    crops = list(global_crops) + list(local_crops)

    num_views = len(crops)
    num_globals = 2

    if max_cols is not None:
        num_views = min(num_views, max_cols)
        crops = crops[:num_views]

    num_locals = max(0, num_views - num_globals)
    B = crops[0].shape[0]
    n_plot = min(n, B)

    # Prep viz tensors (upsample locals)
    viz = []
    for i in range(num_views):
        x = crops[i][:n_plot].detach()
        if i >= num_globals and upsample_locals_to is not None:
            x = F.interpolate(
                x,
                size=(upsample_locals_to, upsample_locals_to),
                mode="bilinear",
                align_corners=False,
            )
        viz.append(x)

    # Build figure
    fig_w = max(2.0 * num_views, 6.0)
    fig_h = max(2.0 * n_plot, 3.0)
    fig, axes = plt.subplots(nrows=n_plot, ncols=num_views, figsize=(fig_w, fig_h))

    if n_plot == 1 and num_views == 1:
        axes = [[axes]]
    elif n_plot == 1:
        axes = axes.reshape(1, -1)
    elif num_views == 1:
        axes = axes.reshape(-1, 1)

    # Column titles
    col_titles = ["G1", "G2"] + [f"L{i}" for i in range(1, num_locals + 1)]
    col_titles = col_titles[:num_views]
    for c in range(num_views):
        axes[0, c].set_title(f"{title_prefix}{col_titles[c]}", fontsize=10)

    # Plot
    for r in range(n_plot):
        for c in range(num_views):
            img = viz[c][r]  # [3,H,W]
            img = _denorm_img(img, denorm_mean, denorm_std)
            img = img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


