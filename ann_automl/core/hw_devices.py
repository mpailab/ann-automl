"""
File contains parameters for different hardware devices,
which can be used for inference.
"""

import os
import sys


class DeviceParams:
    """ Parameters for a device """
    def __init__(self, name, device_type, *, mem_size, bandwidth, tflops_fp32, tflops_fp16=0,
                 tflops_tc_mixed=0, tflops_tc_tf32=0, tflops_tc_fp16=0, tflops_tc_bf16=0, mobile=False):
        """
        Parameters
        ----------
        name: str
            Name of the device
        device_type: str
            Type of the device
        mem_size: float
            Size of the memory in GB
        bandwidth: float
            Bandwidth of the memory in GB/s
        tflops_fp32: float
            Number of floating point operations per second for fp32 in GFLOPS
        tflops_fp16: float
            Number of floating point operations per second for fp16 in GFLOPS
        tflops_tc_mixed: float
            Number of tensor core operations per second for mixed precision in TFLOPS
        tflops_tc_tf32: float
            Number of tensor core operations per second for tf32 precision in TFLOPS
        tflops_tc_fp16: float
            Number of tensor core operations per second for fp16 precision in TFLOPS
        tflops_tc_bf16: float
            Number of tensor core operations per second for bf16 precision in TFLOPS
        mobile: bool
            True if the device is for mobile devices
        """
        self.name = name
        self.device_type = device_type
        self.mem_size = mem_size*2**30
        self.bandwidth = bandwidth*1e9
        self.flops_fp32 = tflops_fp32*1e9
        self.flops_fp16 = max(tflops_fp16, tflops_fp32)*1e9
        self.flops_tc_mixed = max(tflops_tc_mixed, tflops_fp32)*1e9
        self.flops_tc_tf32 = max(tflops_tc_tf32, tflops_fp32)*1e9
        self.flops_tc_fp16 = max(tflops_tc_fp16*1e9, self.flops_tc_mixed)
        self.flops_tc_bf16 = max(tflops_tc_bf16*1e9, self.flops_tc_tf32)
        self.mobile = mobile

    def __repr__(self):
        return f"{self.name} ({self.device_type}, {self.mem_size * 2**-30:.1} GB, {self.flops_fp32 * 1e-9:.1} GFLOPS (fp32), {self.flops_fp16 * 1e-9:.1} GFLOPS (fp16))"

    def as_dict(self):
        """ Return the parameters as a dictionary """
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def save_as_yaml(self, path):
        """ Save the parameters as a YAML file """
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(self.as_dict(), f)

    def save_as_json(self, path):
        """ Save the parameters as a JSON file """
        import json
        with open(path, 'w') as f:
            json.dump(self.as_dict(), f, indent=4)

    @staticmethod
    def load_from_yaml(path):
        """ Load the parameters from a YAML file """
        import yaml
        res = DeviceParams('', '', mem_size=0, bandwidth=0, tflops_fp32=0)
        with open(path, 'r') as f:
            res.__dict__.update(yaml.safe_load(f))

        return res

    @staticmethod
    def load_from_json(path):
        """ Load the parameters from a JSON file """
        import json
        res = DeviceParams('', '', mem_size=0, bandwidth=0, tflops_fp32=0)
        with open(path, 'r') as f:
            res.__dict__.update(json.load(f))

        return res


_device_memory = []


def get_nv_devices_memory() -> []:
    """ Get the list of memory sizes for each GPU (works only for NVIDIA devices) """
    import pynvml
    try:
        pynvml.nvmlInit()
        try:
            res = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                res.append(mem.total)
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        res = []

    return res


def tf_devices_memory():
    """ Get the list of memory sizes for each GPU (works only for NVIDIA devices) """
    global _device_memory
    if not _device_memory:
        # check if cuda is available from tensorflow
        import tensorflow as tf
        if tf.test.is_built_with_cuda():
            _device_memory = get_nv_devices_memory()
        else:  # get cpu memory
            import psutil
            _device_memory = [psutil.virtual_memory().total // 2]  # assume half of the memory is available for training

    return _device_memory


gpus = [
    # NVIDIA GPUs
    # GeForce
    DeviceParams('GeForce RTX 3090', 'GPU', mem_size=24, bandwidth=936, tflops_fp32=35.58,
                 tflops_tc_mixed=284, tflops_tc_tf32=142, tflops_tc_bf16=284),
    DeviceParams('GeForce RTX 3080', 'GPU', mem_size=10, bandwidth=760, tflops_fp32=30,
                 tflops_tc_mixed=240, tflops_tc_tf32=120, tflops_tc_bf16=240),
    DeviceParams('GeForce RTX 3070', 'GPU', mem_size=8, bandwidth=448, tflops_fp32=20,
                 tflops_tc_mixed=160, tflops_tc_tf32=80, tflops_tc_bf16=160),
    DeviceParams('GeForce RTX 3060 Ti', 'GPU', mem_size=8, bandwidth=448, tflops_fp32=16,
                 tflops_tc_mixed=128, tflops_tc_tf32=64, tflops_tc_bf16=128),
    DeviceParams('GeForce RTX 3060', 'GPU', mem_size=6, bandwidth=360, tflops_fp32=12.74,
                 tflops_tc_mixed=96, tflops_tc_tf32=48, tflops_tc_bf16=96),
    DeviceParams('GeForce RTX 2080 Ti', 'GPU', mem_size=11, bandwidth=616, tflops_fp32=13.4, tflops_fp16=27.9,
                 tflops_tc_mixed=108),
    DeviceParams('GeForce RTX 2080', 'GPU', mem_size=8, bandwidth=448, tflops_fp32=11.2, tflops_fp16=22.4,
                 tflops_tc_mixed=89.6),
    DeviceParams('GeForce RTX 1080', 'GPU', mem_size=8, bandwidth=320, tflops_fp32=8.2),
    DeviceParams('GeForce GTX 1080 Ti', 'GPU', mem_size=11, bandwidth=354, tflops_fp32=11.3),

    # Titan
    DeviceParams('TITAN RTX', 'GPU', mem_size=24, bandwidth=672, tflops_fp32=16.3, tflops_fp16=32.6,
                 tflops_tc_mixed=130),
    DeviceParams('TITAN V', 'GPU', mem_size=12, bandwidth=651, tflops_fp32=14.9, tflops_fp16=29.8,
                 tflops_tc_mixed=120, tflops_tc_fp16=120),

    # Tesla
    DeviceParams('Tesla A100 SXM4 40GB', 'GPU', mem_size=40, bandwidth=1555, tflops_fp32=19.49, tflops_fp16=77.96,
                 tflops_tc_mixed=312, tflops_tc_tf32=156, tflops_tc_fp16=312),
    DeviceParams('Tesla A100 SXM4 80GB', 'GPU', mem_size=80, bandwidth=2039, tflops_fp32=19.49, tflops_fp16=77.96,
                 tflops_tc_mixed=312, tflops_tc_tf32=156, tflops_tc_fp16=312),
    DeviceParams('Tesla V100 SXM2 16GB', 'GPU', mem_size=16, bandwidth=900, tflops_fp32=15.3, tflops_fp16=30.7,
                 tflops_tc_mixed=61.4, tflops_tc_fp16=122.8),
    DeviceParams('Tesla V100 SXM2 32GB', 'GPU', mem_size=32, bandwidth=900, tflops_fp32=15.3, tflops_fp16=30.7,
                 tflops_tc_mixed=125),
    DeviceParams('Tesla P100 SXM2', 'GPU', mem_size=16, bandwidth=732, tflops_fp32=10.5, tflops_fp16=21.1)
]
