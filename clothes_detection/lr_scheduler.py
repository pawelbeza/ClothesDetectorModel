import logging

import torch
from detectron2.engine import LRScheduler
from fvcore.common.param_scheduler import LinearParamScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, cfg, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience,
                         threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                         min_lr=min_lr, eps=eps, verbose=verbose)

        start_value = cfg.SOLVER.WARMUP_FACTOR * cfg.SOLVER.BASE_LR
        self._warmup_scheduler = LinearParamScheduler(start_value, cfg.SOLVER.BASE_LR)
        self._cfg = cfg

    def step(self, metrics, epoch=1000):
        if epoch > self._cfg.SOLVER.WARMUP_ITERS:
            super().step(metrics, epoch)
            return

        # warm up
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self._warmup_scheduler(epoch / self._cfg.SOLVER.WARMUP_ITERS)
            param_group['lr'] = new_lr


class LRSchedulerWithLossHook(LRScheduler):
    def __init__(self):
        super().__init__()

    def after_step(self):
        latest = self.trainer.storage.latest()

        if "validation_loss" in latest:
            loss = latest["validation_loss"]
        else:
            loss = 1

        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

        self.scheduler.step(loss, epoch=self.trainer.iter)

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)
