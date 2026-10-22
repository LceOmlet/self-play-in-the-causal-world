"""Select vLLM's own CUDA fallback when NVML detection is unavailable."""


def register():
    import torch
    if not torch.cuda.is_available():
        return None
    # vLLM's CudaPlatform alias itself selects NvmlCudaPlatform or the upstream
    # NonNvmlCudaPlatform. No device method or numerical kernel is replaced.
    return 'vllm.platforms.cuda.CudaPlatform'
