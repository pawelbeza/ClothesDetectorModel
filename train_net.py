from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch, default_argument_parser
from detectron2.data.datasets import register_coco_instances

from clothes_detection.trainer import Trainer

TRAIN_DATASET = "deepfashion_train"
VALIDATION_DATASET = "deepfashion_validation"


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    register_coco_instances(TRAIN_DATASET, {}, args.train_annos, args.train_images)
    register_coco_instances(VALIDATION_DATASET, {}, args.val_annos, args.val_images)

    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def get_parser():
    parser = default_argument_parser()

    parser.add_argument(
        "--train-annos",
        default="./dataset/detectron_annos/train_annos_all.json",
        help="path to json file with training annotations",
    )
    parser.add_argument(
        "--train-images",
        default="./dataset/train/image",
        help="path to dictionary with training images",
    )

    parser.add_argument(
        "--val-annos",
        default="./dataset/detectron_annos/validation_annos_5000.json",
        help="path to json file with validation annotations",
    )
    parser.add_argument(
        "--val-images",
        default="./dataset/validation/image",
        help="path to dictionary with validation images",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
