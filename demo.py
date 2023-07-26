# with open('/data/data_wbw/data/tyre/my_test.json', 'r') as f:
#         data = f.read()
# dicts = json.loads(data)["images"]
# max_len = 0
# import os
# for dic in dicts:
#     file_name = os.path.join('/home/wubw/voc-tire-fbb/testfiles/test_png', dic["file_name"])
#     idx = filename_dict[file_name]
#     ann = dataset.get_ann_info(idx)
#     labels = ann['labels']
#     if len(labels) >= max_len and len(labels) < 10:
#         max_len = len(labels)
#     print(len(labels), file_name)