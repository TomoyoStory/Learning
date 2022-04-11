import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# For ONNX:

class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype = np.float32):
        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)
        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) #^ 这里载入logger和运行时，相关接口可以参考python api
        engine = runtime.deserialize_cuda_engine(f.read()) #^ 从文本二进制的.trt格式的数据中进行ICudaEngine类的获取
        self.context = engine.create_execution_context() #^ 每一个运行的ICudaEngine需要IExecutionContext，及对应的执行上下文

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)] #! 这里需要注意，之所以这里使用int，其根本原因是python是不支持指针的，python中的id本质上返回的就是指针的位置，因此，这里本质上是两个指针的python抽象化表述
        self.stream = cuda.Stream()

    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream) # host to device
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None) #^ execute_async_v2和execute_async的区别在于是否指定batch size
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream) # device to host
        # Syncronize threads
        self.stream.synchronize() #^ host端的流同步

        return self.output

PRECISION = np.float32
BATCH_SIZE=32
N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3))
predictions = trt_model.predict(dummy_input_batch)