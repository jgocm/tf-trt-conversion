# tf-trt-conversion
TF2 to TF-TRT conversion for execution on Jetson Nano

Repository for converting TF2 SSD MobileNet v2 FPNLite 320x320 and 640x640 from [TF2 Models Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to TF-TRT.

Jetson Nano runs out of memory during model conversion as in:
- [Trt_convert converter.convert() gets killed without errors](https://forums.developer.nvidia.com/t/trt-convert-converter-convert-gets-killed-without-errors/148113/6)
- [Converting tf model on jetson tx2 is slow](https://forums.developer.nvidia.com/t/converting-tf-model-on-jetson-tx2-is-slow/125880)
- [Export to TensorRT engine on a Jetson Nano gives 'Killed'](https://github.com/NVIDIA/retinanet-examples/issues/144)

References for conversion script:
- [Tensorflow 2 models on Jetson Nano](https://discuss.tensorflow.org/t/tensorflow-2-models-on-jetson-nano/6122/6)
- [Converting TF 2 Object Detection Model to TensorRT](https://github.com/tensorflow/tensorrt/issues/207#issuecomment-881422008)
- [TF-TRT API In TensorFlow 2.0](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api-20)
- [tf2_inference.py](https://github.com/tensorflow/tensorrt/blob/master/tftrt/blog_posts/Leveraging%20TensorFlow-TensorRT%20integration%20for%20Low%20latency%20Inference/tf2_inference.py)

Things to try:
- Run `$ sudo tegrastats` during execution

- Reduce tf memory limit:
```
tf.config.experimental.set_virtual_device_configuration(
            gpu_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
               memory_limit=512)])
```
- Reduce max_workspace_size_bytes:
> Default value is 1GB. The maximum GPU temporary memory which the TensorRT engine can use at execution time. This corresponds to the workspaceSize parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
```
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<29)) 
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=1)
#conversion_params = conversion_params._replace(use_calibration=True)
```
