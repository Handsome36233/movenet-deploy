# movenet deploy

[MoveNet](https://t.co/QpfnVL0YYI?amp=1) 是一个超快且准确的模型，可检测身体的 17 个关键点。该模型在 [TF Hub](https://tfhub.dev/s?q=movenet) 上提供两种变体，分别为 Lightning 和 Thunder。Lightning 用于延迟关键型应用，而 Thunder 用于需要高准确性的应用。在大多数现代台式机、笔记本电脑和手机上，这两种模型的运行速度都快于实时 (30+ FPS)，这对于实时的健身、健康和保健应用至关重要。参考 [MoveNet：超快且准确的姿态检测模型 &nbsp;|&nbsp; TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/movenet?hl=zh-cn)

将模型从tflite格式转到onnx后，可在各个平台进行部署。

下载onnx格式模型  https://pan.baidu.com/s/17CLEyQTMOaXy5n4oqQ-7AA?pwd=g34u 提取码: g34u

### python

```shell
cd python
python main.py --onnx-model-path --image-path
```

### cpp

```shell
cd cpp
mkdir build && cd build
cmake .. && make
./demo --onnx-model-path --image-path
```

### rust

```shell
cd rust
cargo build --release
./target/release/demo 
```