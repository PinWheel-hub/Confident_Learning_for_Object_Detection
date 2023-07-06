from cleanlab.classification import CleanLearning
from cleanlab.rank import order_label_issues
from cleanlab.filter import find_label_issues
from cleanlab.count import compute_confident_joint

import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction
from mmcv.ops import nms

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

import matplotlib.colors as colors
import re

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap

tail_dot_rgx = re.compile(r'(?:(\.)|(\.\d*?[1-9]\d*?))0+(?=\b|[^0-9])')

def remove_tail_dot_zeros(a):
    return tail_dot_rgx.sub(r'\2',a)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument(
    #     'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        '--save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


cls_num = None
def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.CLASSES)
    global cls_num
    cls_num = np.zeros((num_classes))
    # confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))
    if os.path.exists(os.getcwd() + '/clean/clean_msg.txt'):
        os.remove(os.getcwd() + '/clean/clean_msg.txt')
    if os.path.exists(os.getcwd() + '/clean/clean_msg_fb.txt'):
        os.remove(os.getcwd() + '/clean/clean_msg_fb.txt')
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        analyze_per_img_dets(dataset[idx], gt_bboxes, labels, res_bboxes,
                             num_classes, score_thr, tp_iou_thr)
        prog_bar.update()


def analyze_per_img_dets(data,
                         gt_bboxes,
                         gt_labels,
                         result,
                         class_num,
                         score_thr=0,
                         tp_iou_thr=0.5):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    
    for j, gt_label in enumerate(gt_labels):
        cls_num[gt_label] += 1
     

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir='confusion_matrix.png',
                          show=True,
                          title='confident joint',
                          color_theme='copper'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    confusion_matrix_ = \
        confusion_matrix.astype(np.float32)
    confusion_matrix_[confusion_matrix_!=0] += 5
    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=400)
    cmap = plt.get_cmap(color_theme)
    # cmap = truncate_colormap(cmap, minval=0.1)
    
    im = ax.imshow(confusion_matrix, cmap=cmap)
    cb = plt.colorbar(mappable=im, ax=ax)
    cb.ax.tick_params(labelsize=8)
    im = ax.imshow(confusion_matrix_, cmap=cmap)

    title_font = {'weight': 'bold', 'size': 10}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth', fontdict=label_font)
    plt.xlabel('Prediction', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}'.format(remove_tail_dot_zeros("%.2f" % confusion_matrix[i, j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    dir_name = os.path.dirname(save_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_dir, format='png', bbox_inches='tight')
    if show:
        plt.show()

def main(): 
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # results = mmcv.load(args.prediction_path)
    # assert isinstance(results, list)
    # if isinstance(results[0], list):
    #     pass
    # elif isinstance(results[0], tuple):
    #     results = [result[0] for result in results]
    # else:
    #     raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # calculate_confusion_matrix(dataset, results, score_thr=0.01, tp_iou_thr=0.1)

    # cls_num = np.zeros(len(dataset.CLASSES)).astype(int)
    # # count the instance number in each image
    # for idx in range(len(dataset)):
    #     label = dataset.get_ann_info(idx)['labels']
    #     unique, counts = np.unique(label, return_counts=True)
    #     if len(unique) > 0:
    #         # add the occurrence number to each class
    #         cls_num[unique] += counts

    labels_np = np.load("clean/labels.npy")
    pred_probs_np = np.load("clean/pred_probs.npy")
    labels_fb_np = np.load("clean/labels_fb.npy")
    pred_probs_fb_np = np.load("clean/pred_probs_fb.npy")
    confident_joint = np.load("clean/confident_joint.npy")
    confident_joint_fb = np.load("clean/confident_joint_fb.npy")
    q_joint = confident_joint / np.expand_dims(confident_joint.sum(axis=0), axis=1) * np.expand_dims(confident_joint.sum(axis=1), axis=1) / confident_joint.sum() * 100
    q_joint_fb = confident_joint_fb /  np.expand_dims(confident_joint_fb.sum(axis=0), axis=1) * np.expand_dims(confident_joint_fb.sum(axis=1), axis=1) / confident_joint_fb.sum() * 100

    index = find_label_issues(       
        labels=labels_np,
        pred_probs=pred_probs_np,
        filter_by="confident_learning",
        return_indices_ranked_by="self_confidence",
        frac_noise=1.0,
        verbose=True
    )

    index_fb = find_label_issues(
        labels=labels_fb_np,
        pred_probs=pred_probs_fb_np,
        filter_by="confident_learning",
        return_indices_ranked_by="self_confidence",
        frac_noise=1.0,
        verbose=True
    )

    # result = {}
    # with open("clean/clean_msg.txt", "r") as f:
    #     msg = f.read().splitlines()
    # for idx in index:
    #     if msg[idx].split(" ", 1)[0] not in result:
    #         result[msg[idx].split(" ", 1)[0]] = []
    #         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
    #     else:
    #         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])

    # with open("clean/clean_msg_fb.txt", "r") as f:
    #     msg = f.read().splitlines()
    # for idx in index_fb:
    #     if msg[idx].split(" ", 1)[0] not in result:
    #         result[msg[idx].split(" ", 1)[0]] = []
    #         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
    #     else:
    #         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])

    # with open('clean/my_wrong.txt', 'w') as f:
    #     for key in result:
    #         f.writelines(key + ': ' + ', '.join(result[key]))
    #         f.write('\n')

    plot_confusion_matrix(
        q_joint,
        dataset.CLASSES,
        save_dir='clean/my_data/confident_joint.png',
        show=args.show)

    plot_confusion_matrix(
        q_joint_fb,
        ('FG', 'BG'),
        save_dir='clean/my_data/confident_joint_fb.png',
        show=args.show)

if __name__ == '__main__':
    main()