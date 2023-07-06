from mmcv import Config
import os.path as osp
import mmcv
from mmcv.runner import load_checkpoint
import torch

from mmdet.datasets import build_dataset
from mmdet.apis import inference_detector, show_result_pyplot, show_gt_pyplot, show_result_gt_pyplot
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, update_data_root
import numpy as np

if __name__ == "__main__":


    config_file = 'my_configs/tyre_tood_config.py'
    # Setup a checkpoint file to load
    checkpoint_file = 'checkpoints/my_best.pth'

    # Set the device to be used for evaluation
    device = 'cuda:1'

    # Load the config
    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.test_cfg.nms.iou_threshold = 0.2
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # import json
    # filename_dict = {}
    # with open('clean/my_data/filename_dict.txt', 'w') as f:
    #     for i, data in enumerate(dataset):
    #         filename_dict[data['img_metas'][0].data['filename']] = i
    #     f.write(json.dumps(filename_dict))
    # Initialize the detector
    model = build_detector(cfg.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = cfg

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    filename = '/home/wubw/voc-tire-fbb/testfiles/test_png/I811610852_12341.png'
    img = mmcv.imread(filename)

    result = inference_detector(model, img)


    import json
    with open('clean/my_data/filename_dict.txt', 'r') as f:
        for line in f:
            filename_dict = json.loads(line)
            break
    idx = filename_dict[filename]
    ann = dataset.get_ann_info(idx)
    gt_bboxes = ann['bboxes']
    labels = ann['labels']
    # for i, re in enumerate(result):
    #     index = []
    #     for j, r in enumerate(re):
    #         if r[4] < 0.4:
    #             index.append(j)
    #     re = np.delete(re, index, axis=0)
    #     result[i] = re
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    print(result)
    show_result_gt_pyplot(model=model, img=img, result=result, score_thr=0.3, bboxes=gt_bboxes, labels=labels)
    # show_result_pyplot(model, img, result, score_thr=0.1)
