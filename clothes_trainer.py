import os

from loss_eval_hook import LossEvalHook
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T


class ClothesTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomApply(T.RandomCrop("absolute", (640, 640)), prob=0.05),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomApply(T.RandomRotation([-10, 10]), prob=0.4),
            T.RandomApply(T.RandomSaturation(0.8, 1.2), prob=0.3),
            T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.2),
            T.RandomApply(T.RandomContrast(0.6, 1.3), prob=0.2),
            T.RandomApply(T.RandomLighting(0.7), prob=0.4),
        ]))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks
