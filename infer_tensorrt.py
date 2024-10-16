# -- coding: utf-8 --
# @Time : 2024/10/16 11:07
# @Author : Zeng Li


from onnx_helper import ONNXClassifierWrapper
import numpy as np
import time

# 精度
PRECISION = np.float32
# PRECISION = np.float16
# PRECISION = np.int8

# The input tensor shape of the ONNX model.
input_shape = (1, 2, 128, 128)

dummy_input_batch = np.zeros(input_shape, dtype=PRECISION)

# engine_name = "ResNet10.engine"
# engine_name = "ResNet18.engine"
# engine_name = "ResNet50.engine"
# engine_name = "ResNet10_PE128.engine"
# engine_name = "ResNet18_PE128.engine"
# engine_name = "ResNet50_PE128.engine"
# engine_name = "ResNet10_PE64.engine"
# engine_name = "ResNet18_PE64.engine"
engine_name = "ResNet50_PE64.engine"

trt_model = ONNXClassifierWrapper(engine_name, target_dtype = PRECISION)

# 进行多次推理以获取平均时间
num_warmup_runs = 500
num_inference_runs = 1000

out = trt_model.predict(dummy_input_batch) # softmax probability predictions for the first 10 classes of the first sample
print(out.shape)
print(out)

# Warm-up runs
for _ in range(num_warmup_runs):
    out = trt_model.predict(dummy_input_batch) # softmax probability predictions for the first 10 classes of the first sample

# Measure inference time
total_time = 0
for _ in range(num_inference_runs):
    start_time = time.perf_counter()
    trt_model.predict(dummy_input_batch)
    end_time = time.perf_counter()
    total_time += (end_time - start_time)

average_time_per_inference = total_time / num_inference_runs
print(f"Average inference time per run: {average_time_per_inference:.6f} seconds")
FPS = 1.0 / average_time_per_inference
print(f"Average FPS: {FPS:.6f} ")


# Measure inference time
total_time = 0
for _ in range(num_inference_runs):
    start_time = time.time()
    trt_model.predict(dummy_input_batch)
    end_time = time.time()
    total_time += (end_time - start_time)

average_time_per_inference = total_time / num_inference_runs
print(f"Average inference time per run: {average_time_per_inference:.6f} seconds")
FPS = 1.0 / average_time_per_inference
print(f"Average FPS: {FPS:.6f} ")