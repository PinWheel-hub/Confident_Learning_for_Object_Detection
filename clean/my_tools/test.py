# with open("clean_result.txt", "r") as f:
#     index = f.read().splitlines()
# index = [int(idx) for idx in index]
# with open("clean_msg.txt", "r") as f:
#     msg = f.read().splitlines()
# result = {}
# for idx in index:
#     if msg[idx].split(" ", 1)[0] not in result:
#         result[msg[idx].split(" ", 1)[0]] = []
#         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
#     else:
#         result[msg[idx].split(" ", 1)[0]].append(msg[idx].split(" ", 1)[1])
# with open('wrong_pictures.txt', 'w') as f:
#     for key in result:
#         f.writelines(key + ': ' + ', '.join(result[key]))
#         f.write('\n')

# for idx, data in enumerate(dataset):
    #     filename_dict[data['img_metas'][0].data['filename']] = idx
    # with open('clean/filename_dict.txt', 'w') as f:
    #     f.write(json.dumps(filename_dict))
# def analyze_per_img_dets(data,
#                          confusion_matrix,
#                          gt_bboxes,
#                          gt_labels,
#                          result,
#                          score_thr=0,
#                          tp_iou_thr=0.5,
#                          nms_iou_thr=None):
#     """Analyze detection results on each image.
#
#     Args:
#         confusion_matrix (ndarray): The confusion matrix,
#             has shape (num_classes + 1, num_classes + 1).
#         gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
#         gt_labels (ndarray): Ground truth labels, has shape (num_gt).
#         result (ndarray): Detection results, has shape
#             (num_classes, num_bboxes, 5).
#         score_thr (float): Score threshold to filter bboxes.
#             Default: 0.
#         tp_iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         nms_iou_thr (float|optional): nms IoU threshold, the detection results
#             have done nms in the detector, only applied when users want to
#             change the nms IoU threshold. Default: None.
#     """
#     true_positives = np.zeros_like(gt_labels)
#     positives = np.zeros_like(gt_labels)
#     max_scores = np.zeros(len(gt_labels))
#     max_probs = [0] * len(gt_labels)
#     max_labels = [-1] * len(gt_labels)
#     class_num = 13
#     with open("clean/clean_msg.txt", "a") as f:
#         for det_label, det_bboxes in enumerate(result):
#             if nms_iou_thr:
#                 det_bboxes, _ = nms(
#                     det_bboxes[:, :4],
#                     det_bboxes[:, -1],
#                     nms_iou_thr,
#                     score_threshold=score_thr)
#             ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
#             for i, det_bbox in enumerate(det_bboxes):
#                 score = np.max(det_bbox[4:])
#                 det_match = 0
#                 rank = np.argsort(ious[i, :])[::-1]
#                 for j in rank:
#                     if ious[i, j] < tp_iou_thr:
#                         break
#                     if true_positives[j] > 0 and gt_labels[j] != det_label:
#                         det_match += 1
#                         break
#                     if score > max_scores[j] and (det_label == gt_labels[j] or max_labels[j] != gt_labels[j]):
#                         max_scores[j] = score
#                         max_probs[j] = det_bbox[4:]
#                         max_labels[j] = det_label
#                     if score > score_thr:
#                         msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], det_label,
#                                                                                                 gt_bboxes[j], score)
#                         f.write(msg + '\n')
#                         pred_prob = np.append(det_bbox[4:], max(0.6 - np.sum(det_bbox[4:]), 0))
#                         pred_probs.append(pred_prob)
#                         labels.append(gt_labels[j])
#                         det_match += 1
#                         positives[j] += 1
#                         if gt_labels[j] == det_label:
#                             true_positives[j] += 1  # TP
#                         # confusion_matrix[gt_label, det_label] += 1
#                         break
#                 if det_match == 0 and score > score_thr:
#                     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(class_num, det_label, det_bbox[:4], score)
#                     f.write(msg + '\n')
#                     pred_prob = np.append(det_bbox[4:], max(0.6 - np.sum(det_bbox[4:]), 0))
#                     pred_probs.append(pred_prob)
#                     labels.append(class_num)
#                 # if det_match == 0:  # BG FP
#                 #     filenames.append(data['img_metas'][0].data['filename'])
#                 #     pred_probs.append(np.append(det_bbox[4:], 1 - np.sum(det_bbox[4:])))
#                 #     labels.append(len(det_bbox) - 4)
#                 #     confusion_matrix[-1, det_label] += 1
#         for j, gt_label in enumerate(gt_labels):
#             if positives[j] == 0:  # FN
#                 if max_scores[j] > 0.05:
#                     if max_scores[j] < 0.6 - np.sum(max_probs[j]):
#                         msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_label, class_num, gt_bboxes[j], 0.6 - np.sum(max_probs[j]))
#                     else:
#                         msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_label, max_labels[j], gt_bboxes[j], max_scores[j])
#                     f.write(msg + '\n')
#                     pred_prob = np.append(max_probs[j], max(0.6 - np.sum(max_probs[j]), 0))
#                     pred_probs.append(pred_prob)
#                     labels.append(gt_label)
#                 else:
#                     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_label, class_num, gt_bboxes[j], 0.6)
#                     f.write(msg + '\n')
#                     # print(np.array([0.] * class_num + [1]))
#                     pred_prob = np.array([0.] * class_num + [0.6])
#                     pred_probs.append(pred_prob)
#                     labels.append(gt_label)
#                 # confusion_matrix[gt_label, -1] += 1

# def analyze_per_img_dets(data,
#                          confusion_matrix,
#                          gt_bboxes,
#                          gt_labels,
#                          result,
#                          score_thr=0,
#                          tp_iou_thr=0.5,
#                          nms_iou_thr=None):
#     """Analyze detection results on each image.

#     Args:
#         confusion_matrix (ndarray): The confusion matrix,
#             has shape (num_classes + 1, num_classes + 1).
#         gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
#         gt_labels (ndarray): Ground truth labels, has shape (num_gt).
#         result (ndarray): Detection results, has shape
#             (num_classes, num_bboxes, 5).
#         score_thr (float): Score threshold to filter bboxes.
#             Default: 0.
#         tp_iou_thr (float): IoU threshold to be considered as matched.
#             Default: 0.5.
#         nms_iou_thr (float|optional): nms IoU threshold, the detection results
#             have done nms in the detector, only applied when users want to
#             change the nms IoU threshold. Default: None.
#     """
#     # true_positives = np.zeros_like(gt_labels)
#     # positives = np.zeros_like(gt_labels)
#     # max_scores = np.zeros(len(gt_labels))
#     # max_probs = [0] * len(gt_labels)
#     # max_labels = [-1] * len(gt_labels)
#     class_num = 13
#     det_labels = []
#     det_results = []
#     scores = []
#     det_num = 0
#     for det_label, det_bboxes in enumerate(result):
#         for i, det_bbox in enumerate(det_bboxes):
#             scores.append(np.max(det_bbox[4:]))
#             det_labels.append(det_label)
#             det_results.append(det_bbox)
#             det_num += 1
#     det_results = np.array(det_results)
#     positives = np.zeros(det_num)
#     with open("clean/clean_msg.txt", "a") as f:
#         for j, gt_bbox in enumerate(gt_bboxes):
#             max_score = 0
#             max_prob = 0
#             max_label = -1
#             det_match = 0
#             ious = bbox_overlaps(det_results[:, :4], gt_bboxes)
#             rank = np.argsort(ious[:, j])[::-1]
#             for i in rank:
#                 if ious[i, j] < tp_iou_thr:
#                     break
#                 if scores[i] > max_score and (det_labels[i] == gt_labels[j] or max_label != gt_labels[j]):
#                     max_score = scores[i]
#                     max_prob = det_results[i][4:]
#                     max_label = det_labels[i]
#                 if scores[i] > score_thr:
#                     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], det_labels[i],
#                                                                                             gt_bbox, scores[i])
#                     f.write(msg + '\n')
#                     pred_prob = np.append(det_results[i][4:], max(0.6 - np.sum(det_results[i][4:]), 0))
#                     pred_probs.append(pred_prob)
#                     labels.append(gt_labels[j])
#                     positives[i] += 1
#                     det_match += 1
#                     break
#             if det_match == 0:
#                 if max_score > 0:
#                     if max_score < 0.6 - np.sum(max_prob):
#                         msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], class_num, gt_bboxes[j], 0.6 - np.sum(max_prob))
#                     else:
#                         msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], max_label, gt_bboxes[j], max_score)
#                     f.write(msg + '\n')
#                     pred_prob = np.append(max_prob, max(0.6 - np.sum(max_prob), 0))
#                     pred_probs.append(pred_prob)
#                     labels.append(gt_labels[j])
#                 else:
#                     msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(gt_labels[j], class_num,
#                                                                                             gt_bboxes[j], 0.6)
#                     f.write(msg + '\n')
#                     pred_prob = np.array([0.] * class_num + [0.6])
#                     pred_probs.append(pred_prob)
#                     labels.append(gt_labels[j])

#         for i, det_result in enumerate(det_results):
#             if positives[i] == 0 and scores[i] > score_thr:  # FN
#                 msg = data['img_metas'][0].data['filename'] + ' {0} {1} {2} {3}'.format(class_num, det_labels[i],
#                                                                                         det_result[:4], scores[i])
#                 f.write(msg + '\n')
#                 pred_prob = np.append(det_result[4:], max(0.6 - np.sum(det_result[4:]), 0))
#                 pred_probs.append(pred_prob)
#                 labels.append(class_num)

if __name__ == '__main__':
    wrong_dict = {}
    with open("../my_wrong_k.txt", 'r') as f:
        msgs = f.read().splitlines()
    for msg in msgs:
        wrong_dict[msg.split(" ", 1)[0].split("/")[-1].split(".", 1)[0]] = True
    with open("/data/data_wbw/data/tyre/my_train.txt", "r") as f:
        trains = f.read().splitlines()
    with open("/data/data_wbw/data/tyre/my_train_cleaned.txt", "w") as f:
        for train in trains:
            if train not in wrong_dict:
                f.write(train + "\n")