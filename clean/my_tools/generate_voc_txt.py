import os
data_root = '/home/wubw/voc-tire-fbb/'
with open('/data/data_wbw/data/tyre/train.txt', 'w') as f:
    for root_dir, _, files in os.walk(data_root + 'trainfiles/train_png'):
        for file in files:
            file_name = os.path.join(root_dir, file)
            f.write(file[:-4])
            f.write('\n')
            print(file[:-4])
with open('/data/data_wbw/data/tyre/test.txt', 'w') as f:
    for root_dir, _, files in os.walk(data_root + 'testfiles/test_png'):
        for file in files:
            file_name = os.path.join(root_dir, file)
            f.write(file[:-4])
            f.write('\n')
            print(file[:-4])