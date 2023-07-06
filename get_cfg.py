from mmcv import Config
import os.path as osp
import mmcv
from mmcv.runner import build_runner, load_checkpoint
import torch

from mmdet.apis import inference_detector, show_result_pyplot, init_detector, set_random_seed, train_detector
from mmdet.utils import get_root_logger
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

if __name__ == "__main__":
    config_file = 'configs/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco.py'
    # Setup a checkpoint file to load
    checkpoint_file = 'checkpoints/resnet50-0676ba61.pth'

    # Load the config
    cfg = mmcv.Config.fromfile(config_file)

    # # Set pretrained to be None since we do not need pretrained model here
    # cfg.model.bbox_head.num_classes = 6
    # cfg.model.bbox_head.loss_bbox.type = 'IoULoss'
    # cfg.model.bbox_head.loss_bbox.loss_weight = 0.1
    #
    # # Initialize the detector
    # cfg.model.pretrained = None
    # model = build_detector(cfg.model)
    #
    # # Load checkpoint
    #
    # load_checkpoint(model, checkpoint_file)
    #
    # # Modify dataset type and path
    # cfg.dataset_type = 'COCODataset'
    #
    # cfg.data.test.ann_file = '../data/neu_det_coco/annotations/test_annotation.json'
    # cfg.data.test.img_prefix = '../data/neu_det_coco/test/'
    # cfg.data.test.classes = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
    #
    # cfg.data.train.ann_file = '../data/neu_det_coco/annotations/train_annotation.json'
    # cfg.data.train.img_prefix = '../data/neu_det_coco/train/'
    # cfg.data.train.classes = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
    #
    #
    # cfg.data.val.ann_file = '../data/neu_det_coco/annotations/test_annotation.json'
    # cfg.data.val.img_prefix = '../data/neu_det_coco/test/'
    # cfg.data.val.classes = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")


    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    # cfg.optimizer.lr = 0.02 / 80
    # cfg.lr_config.warmup = None
    # cfg.log_config.interval = 50
    #
    # # We can set the evaluation interval to reduce the evaluation times
    # cfg.evaluation.interval = 2
    # # We can set the checkpoint saving interval to reduce the storage cost
    # cfg.checkpoint_config.interval = 2

    # Set seed thus the results are more reproducible
    cfg.seed = 10086
    # set_random_seed(cfg.seed, deterministic=False)
    # cfg.gpu_ids = range(1)

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]

    # cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg.runner.max_epochs = 40
    # cfg.data.samples_per_gpu = 6
    # cfg.data.workers_per_gpu = 2
    #
    # cfg.test_pipeline[1].img_scale = (400, 400)
    # cfg.train_pipeline[2].img_scale = (400, 400)
    # cfg.data.test.pipeline[1].img_scale = (400, 400)
    # cfg.data.train.pipeline[2].img_scale = (400, 400)
    # cfg.data.val.pipeline[1].img_scale = (400, 400)
    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # Build dataset
    # datasets = [build_dataset(cfg.data.train)]
    # # Build the detector
    #
    # # Add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    # print(model)
    # # Create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # train_detector(model, datasets, cfg, distributed=False, validate=True)


    # img = mmcv.imread('../data/neu_det_coco/test/crazing_67.jpg')
    # model.cfg = cfg
    # result = inference_detector(model, img)
    # show_result_pyplot(model, img, result)
