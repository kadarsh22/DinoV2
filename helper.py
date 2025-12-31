import numpy as np
from torch.optim import AdamW


def init_optimizers(model, optim_config, batch_size, train_loader):
    optimizer = AdamW(model.parameters(), lr=optim_config['lr'])
    lr_scheduler = cosine_scheduler(optim_config['lr'] * (batch_size / 256.),
                                   optim_config['min_lr'], optim_config['epochs'], len(train_loader),
                                   warmup_epochs=optim_config['warmup_epochs'])

    wd_schedule = cosine_scheduler(optim_config['wd_start'], optim_config['wd_end'], optim_config['epochs'], len(train_loader),)
    return optimizer, lr_scheduler, wd_schedule


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule