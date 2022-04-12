#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# NVIDIA 官网原始案例，helper.py已集成进来

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import tensorrt as trt
import numpy as np
import timeit

precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

# For TF-TRT:
class OptimizedModel():
    def __init__(self, saved_model_dir = None):
        self.loaded_model_fn = None

        if not saved_model_dir is None:
            self.load_model(saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model_fn is None:
            raise(Exception("Haven't loaded a model"))
        x = tf.constant(input_data.astype('float32'))
        labeling = self.loaded_model_fn(x)
        try:
            preds = labeling['predictions'].numpy()
        except:
            try:
                preds = labeling['probs'].numpy()
            except:
                try:
                    preds = labeling[next(iter(labeling.keys()))]
                except:
                    raise(Exception("Failed to get predictions from saved model object"))
        return preds

    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING]) #^ SavedModel包含一个或多个模型变体(技术为 v1.MetaGraphDef)，这些变体通过tag-set进行标识。因此，这里必须选择对应的tag，单独保存的模型tag默认就是tag_constants.SERVING
        wrapper_fp32 = saved_model_loaded.signatures['serving_default'] #^ SaveModel的签名请详细参考Tensorflow官网，其实相当于指定图的节点输入位置，然后predict得到图的输出
        self.loaded_model_fn = wrapper_fp32

class ModelOptimizer():
    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        self.loaded_model = None
        if not calibration_data is None:
            self.set_calibration_data(calibration_data)

    def set_calibration_data(self, calibration_data):
        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype('float32')), )
        self.calibration_data = calibration_input_fn

    def convert(self, output_saved_model_dir, precision="FP32", max_workspace_size_bytes=8000000000, **kwargs):

        if precision == "INT8" and self.calibration_data is None:
            raise(Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision,
                                                                       max_workspace_size_bytes=max_workspace_size_bytes,
                                                                       use_calibration= precision == "INT8")
        converter = tf_trt.TrtGraphConverterV2(input_saved_model_dir=self.input_saved_model_dir,
                                conversion_params=conversion_params)

        if precision == "INT8":
            converter.convert(calibration_input_fn=self.calibration_data)
        else:
            converter.convert()
        converter.save(output_saved_model_dir=output_saved_model_dir)
        return OptimizedModel(output_saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model is None:
            self.load_default_model()
        return self.loaded_model.predict(input_data)

    def load_default_model(self):
        self.loaded_model = tf.keras.models.load_model('resnet50_saved_model')

#~ 详细位置参考 https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb
#~ 以及 https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/quickstart/IntroNotebooks/helper.py
#* 该示例根据TF_TRT示例进行改进得到

model_dir = 'tmp_savedmodels/resnet50_saved_model'
model = ResNet50(include_top=True, weights='imagenet')
model.save(model_dir)
BATCH_SIZE = 32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3))
PRECISION = "FP32" # Options are "FP32", "FP16", or "INT8"
model_dir = 'tmp_savedmodels/resnet50_saved_model'
opt_model = ModelOptimizer(model_dir)
model_fp32 = opt_model.convert(model_dir+'_FP32', precision=PRECISION)
model_fp32.predict(dummy_input_batch)

# Warm up - the first batch through a model generally takes longer
model.predict(dummy_input_batch)
model_fp32.predict(dummy_input_batch)

# timeit
print(timeit.timeit('model.predict_on_batch(dummy_input_batch)')) #^ 看执行1000000次模型推理的时间
print(timeit.timeit('model_fp32.predict(dummy_input_batch)'))

# FLOAT16
model_fp16 = opt_model.convert(model_dir+'_FP16', precision="FP16")
model_fp16.predict(dummy_input_batch)
print(timeit.timeit('model_fp16.predict(dummy_input_batch)'))

# INT8
#^ 在图灵架构之后的显卡，其INT8的计算能力较好， Jetson AGX Xavier和T4这种卡INT8的效果较好(专注于板卡)，而专注于训练的A100其性能和float16差别不大
dummy_calibration_batch = np.zeros((8, 224, 224, 3))
opt_model.set_calibration_data(dummy_calibration_batch)
model_int8 = opt_model.convert(model_dir+'_INT8', precision="INT8")
model_int8.predict(dummy_input_batch)
print(timeit.timeit('model_int8.predict(dummy_input_batch)'))