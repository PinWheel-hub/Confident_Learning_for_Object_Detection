# Confident_Learning_for_Object_Detection
一种使用置信学习(Confident Learning)清洗目标检测数据集中噪声标签的后处理方法，基于mmdetection框架。

用法示例：
```
python clean/clean.py my_configs/tyre_tood_json_config.py clean/result.pkl 
```
`my_configs/tyre_tood_json_config.py`: 目标检测模型配置文件

`clean/result.pkl`: 模型推理结果，在测试中得到，指令如下
```
python tools/test.py my_configs/tyre_tood_json_config.py tutorial_exps/latest.pth --eval bbox --out clean/result.pkl
```

## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

Thanks to the work [Confident Learning](https://github.com/cleanlab/cleanlab).
