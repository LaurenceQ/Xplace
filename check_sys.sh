#!/bin/bash

echo "========== 硬件信息 =========="

# 查看 GPU 型号
echo -e "\n>>> 显示设备:"
lspci | grep -iE "vga|3d|display"

# 如果安装了 NVIDIA 驱动，可以取消下一行的注释来查看更多 GPU 细节
# echo -e "\n>>> NVIDIA 详细信息:"
# nvidia-smi

# 查看 CPU 信息
echo -e "\n>>> CPU 型号及架构:"
lscpu | grep -E "Architecture|Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core"

# 查看内存总大小
echo -e "\n>>> 内存大小及使用情况:"
free -h

echo -e "\n========== 软件环境 =========="
echo -e "\n>>> Python 及 PyTorch/CUDA 信息:"
python3 -c "
import torch
print(f'Python 版本: {__import__("sys").version}')
print(f'PyTorch 版本: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA 编译版本: {torch.version.cuda}')
    print(f'当前 GPU 名称: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA 当前不可用')
"
