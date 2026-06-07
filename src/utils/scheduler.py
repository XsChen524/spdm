import math

from torch import optim
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "sigmoid":
        k, s, w = 0.5, 10, 10
        lr = args.learning_rate / (
            1 + math.exp(-k * (epoch - w))
        ) - args.learning_rate / (1 + math.exp(-k / s * (epoch - w * s)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return
    else:
        return
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def build_scheduler(optimizer, args, train_steps=None):
    scheduler_type = getattr(args, "lradj", "cosine")

    if scheduler_type in ("type1", "type2"):
        return None

    if scheduler_type == "cosine":
        warmup_epochs = getattr(args, "warmup_epochs", 3)
        decay_epochs = max(args.train_epochs - warmup_epochs, 1)
        eta_min = getattr(args, "cosine_eta_min", 1e-7)

        def warmup_fn(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)

        warmup = LambdaLR(optimizer, warmup_fn)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=eta_min)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    if scheduler_type == "onecycle":
        pct_start = getattr(args, "pct_start", 0.1)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.train_epochs,
            steps_per_epoch=train_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

    if scheduler_type == "sigmoid":
        return None  # handled in adjust_learning_rate

    return None
