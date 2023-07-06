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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
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
    # confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))
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


pred_probs = []
labels = []
pred_probs_fb = []
labels_fb = []
bgfg = np.zeros((2))
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
    # true_positives = np.zeros_like(gt_labels)
    # positives = np.zeros_like(gt_labels)
    # max_scores = np.zeros(len(gt_labels))
    # max_probs = [0] * len(gt_labels)
    # max_labels = [-1] * len(gt_labels)
    set_score = 0.96 # 0.95
    det_labels = []
    det_results = []
    scores = []
    det_num = 0
    for det_label, det_bboxes in enumerate(result):
        for i, det_bbox in enumerate(det_bboxes):
            scores.append(np.max(det_bbox[4:]))
            det_labels.append(det_label)
            det_results.append(det_bbox)
            det_num += 1
    det_results = np.array(det_results)
    positives = np.zeros(det_num)
    ious = bbox_overlaps(det_results[:, :4], gt_bboxes)
    with open("clean/clean_msg.txt", "a") as f, open("clean/clean_msg_fb.txt", "a") as f_fb:
        for j, gt_bbox in enumerate(gt_bboxes):
            max_score = 0
            max_prob = 0
            max_label = -1
            max_idx = 0
            det_match = 0     
            rank = np.argsort(ious[:, j])[::-1]
            for i in rank:
                if ious[i, j] < tp_iou_thr:
                    break
                if (scores[i] * scores[i] * ious[i, j] > max_score *  max_score * ious[max_idx, j] and (det_labels[i] == gt_labels[j] or max_label != gt_labels[j])) or (det_labels[i] == gt_labels[j] and max_label != gt_labels[j] and scores[i] > 0.3):
                    max_idx = i
                    max_score = scores[i]
                    max_prob = det_results[i][4:]
                    max_label = det_labels[i]
                # if scores[i] > score_thr:
                #     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], det_labels[i],
                #                                                                             gt_bbox, scores[i])
                #     f.write(msg + '\n')
                #     pred_prob = np.append(det_results[i][4:], max(set_score - np.sum(det_results[i][4:]), 0))
                #     pred_probs.append(pred_prob)
                #     labels.append(gt_labels[j])
                #     positives[i] += 1
                #     det_match += 1
                #     break
            # if det_match == 0:
            if max_score > 0:
                positives[max_idx] += 1
                # if max_score < set_score * np.prod(1 - max_prob):
                #     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], class_num, gt_bboxes[j], set_score * np.prod(1 - max_prob))
                # else:
                #     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], max_label, gt_bboxes[j], max_score)
                msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], max_label,
                                                                                        gt_bboxes[j], max_score)
                f.write(msg + '\n')
                pred_prob = max_prob
                pred_probs.append(pred_prob)
                labels.append(gt_labels[j])

                prob_b = np.prod(1 - max_prob)
                prob_f = 1 - prob_b
                prob_b = set_score * prob_b
                if prob_f >= prob_b:
                    msg_fb = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format('F', 'F', gt_bboxes[j], prob_f)
                else:
                    msg_fb = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format('F', 'B', gt_bboxes[j], prob_f)
                f_fb.write(msg_fb + '\n')
                pred_probs_fb.append(np.array([prob_f, prob_b]))
                labels_fb.append(0)
                bgfg[0] += 1
            else:
                # msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], class_num,
                #                                                                         gt_bboxes[j], set_score)
                # f.write(msg + '\n')
                # pred_prob = np.array([0.] * class_num + [set_score])
                # pred_probs.append(pred_prob)
                # labels.append(gt_labels[j])
                prob_b = set_score
                prob_f = 0
                msg_fb = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format('F', 'B', gt_bboxes[j], prob_f)
                f_fb.write(msg_fb + '\n')
                pred_probs_fb.append(np.array([prob_f, prob_b]))
                labels_fb.append(0)
                bgfg[0] += 1
        for i, det_result in enumerate(det_results):
            if positives[i] == 0 and scores[i] > score_thr:  # FN
                # msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(class_num, det_labels[i],
                #                                                                         det_result[:4], scores[i])
                # f.write(msg + '\n')
                # pred_prob = np.append(det_result[4:], set_score * np.prod(1 - det_result[4:]))
                # pred_probs.append(pred_prob)
                # labels.append(class_num)
                if ious[i,:].size > 0 and np.max(ious[i,:]) > tp_iou_thr:
                    # if gt_labels[np.argmax(ious[i,:])] == det_labels[i]:
                    continue
                prob_b = np.prod(1 - det_result[4:])
                prob_f = (1 - prob_b) * 0.8 # 0.75
                if prob_f >= prob_b:
                    msg_fb = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format('B', 'F', det_result[:4], prob_f)
                else:
                    msg_fb = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format('B', 'B', det_result[:4], prob_f)
                f_fb.write(msg_fb + '\n')
                pred_probs_fb.append(np.array([prob_f, prob_b]))
                labels_fb.append(1)
                bgfg[1] += 1

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir='confusion_matrix.png',
                          show=True,
                          title='confident joint',
                          color_theme='plasma'):
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
    label_sums = confusion_matrix.sum()
    confusion_matrix_ = \
        confusion_matrix.astype(np.float32) / label_sums

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix_, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
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
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

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
                '{}'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
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

    if os.path.exists(os.getcwd() + '/clean/clean_msg.txt'):
        os.remove(os.getcwd() + '/clean/clean_msg.txt')
    if os.path.exists(os.getcwd() + '/clean/clean_msg_fb.txt'):
        os.remove(os.getcwd() + '/clean/clean_msg_fb.txt')

    for i in range(1, 5 + 1):
        results = mmcv.load('clean/my_{}.pkl'.format(i))
        assert isinstance(results, list)
        if isinstance(results[0], list):
            pass
        elif isinstance(results[0], tuple):
            results = [result[0] for result in results]
        else:
            raise TypeError('invalid type of prediction results')

        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
        cfg.data.test.ann_file = '/data/data_wbw/data/tyre/my_val_{}.json'.format(i),
        dataset = build_dataset(cfg.data.test)
        calculate_confusion_matrix(dataset, results, score_thr=0.01, tp_iou_thr=0.1)

    labels_np = np.array(labels)
    np.save('clean/labels.npy', labels_np)

    pred_probs_np = np.array(pred_probs)
    np.save('clean/pred_probs.npy', pred_probs_np)

    labels_fb_np = np.array(labels_fb)
    np.save('clean/labels_fb.npy', labels_fb_np)

    pred_probs_fb_np = np.array(pred_probs_fb)
    np.save('clean/pred_probs_fb.npy', pred_probs_fb_np)

    confident_joint = compute_confident_joint(labels=labels_np,
                                              pred_probs=pred_probs_np,
                                              multi_label=False,
                                              return_indices_of_off_diagonals=False)
    np.save('clean/confident_joint.npy', confident_joint)

    confident_joint_fb = compute_confident_joint(labels=labels_fb_np,
                                              pred_probs=pred_probs_fb_np,
                                              multi_label=False,
                                              return_indices_of_off_diagonals=False)
    np.save('clean/confident_joint_fb.npy', confident_joint_fb)

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

    result = {}
    with open("clean/clean_msg.txt", "r") as f:
        msg = f.read().splitlines()
    for idx in index:
        if msg[idx].split(" ", 1)[0] not in result:
            result[msg[idx].split(" ", 1)[0]] = []
            result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
        else:
            result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])

    with open("clean/clean_msg_fb.txt", "r") as f:
        msg = f.read().splitlines()
    for idx in index_fb:
        if msg[idx].split(" ", 1)[0] not in result:
            result[msg[idx].split(" ", 1)[0]] = []
            result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
        else:
            result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])

    with open('clean/my_wrong_k.txt', 'w') as f:
        for key in result:
            f.writelines(key + ': ' + ', '.join(result[key]))
            f.write('\n')

    # plot_confusion_matrix(
    #     confident_joint,
    #     dataset.CLASSES,
    #     save_dir='clean/confident_joint.png',
    #     show=args.show,
    #     color_theme=args.color_theme)

    # plot_confusion_matrix(
    #     confident_joint_fb,
    #     ('FG', 'BG'),
    #     save_dir='clean/confident_joint_fb.png',
    #     show=args.show,
    #     color_theme=args.color_theme)


if __name__ == '__main__':
    main()
