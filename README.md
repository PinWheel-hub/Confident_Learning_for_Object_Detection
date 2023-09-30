# Confident_Learning_for_Object_Detection
一种使用置信学习(Confident Learning)清洗目标检测数据集中噪声标签的后处理方法，基于mmdetection框架。

用法示例：
```
python clean/clean.py my_configs/tyre_tood_json_config.py clean/result.pkl 
```
## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

Thanks to the work [Confident Learning](https://github.com/cleanlab/cleanlab).