"""GPU selection helper used by the inference scripts."""
import os
import traceback

import nvgpu
import torch


def select_gpu(i_selected_gpu=None):
    """Pick a GPU with free memory and set CUDA_VISIBLE_DEVICES accordingly.

    If `i_selected_gpu` is given, just sets that one. Otherwise, scans all
    visible GPUs (in reverse order) and picks the first one with mem_used < 18
    MiB; if none is free, falls back to the GPU with the most unused memory.
    """
    if i_selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{i_selected_gpu}"
        print(f"CUDA_VISIBLE_DEVICES={i_selected_gpu}")
        return

    unused_max = 0
    is_free_gpu = False
    gpu_info = []
    try:
        gpu_info = nvgpu.gpu_info()
    except BaseException:
        traceback.print_exc()

    unused_i_gpu = 0
    for i_gpu, gpu in reversed(list(enumerate(gpu_info))):
        unused = gpu["mem_total"] - gpu["mem_used"]
        if unused > unused_max:
            unused_i_gpu = i_gpu
            unused_max = unused
        # use this gpu
        if gpu["mem_used"] < 18:
            i_selected_gpu = i_gpu
            is_free_gpu = True
            break
    # There is no free GPU, use less used one.
    if i_selected_gpu is None:
        i_selected_gpu = unused_i_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{i_selected_gpu}"
    print(f"CUDA_VISIBLE_DEVICES={i_selected_gpu}, is_free:{is_free_gpu}")
    # using flag
    if is_free_gpu:
        torch.zeros(2 * 10**4, dtype=torch.float64).cuda()
