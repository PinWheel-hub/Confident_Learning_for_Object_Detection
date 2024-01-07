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
import argparse

if __name__ == '__main__':
    # 创建 ArgumentParser 实例
    parser = argparse.ArgumentParser(description='Description of your program')

    # 添加命令行参数
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    # 解析命令行参数
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Error: {e}")
        # 进行适当的错误处理或提示用户

    # 访问参数值
    input_path = args.input
    output_path = args.output
    verbose_mode = args.verbose

    # 打印参数值
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    print(f'Verbose mode: {verbose_mode}')