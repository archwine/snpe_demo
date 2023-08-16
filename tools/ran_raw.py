import numpy as np
import os

# 创建文件夹用于存储.raw文件
if not os.path.exists("random_arrays"):
    os.mkdir("random_arrays")

for i in range(10):
    # 随机生成3x224x224的数组，数值范围在0到255之间（代表8位无符号整数）
    random_array = np.random.rand(3, 224, 224).astype(np.float32)

    # 保存为.raw格式文件
    raw_filename = os.path.join("random_arrays", f"random_array_{i}.raw")
    random_array.tofile(raw_filename)

    print(f"Random array {i} saved as {raw_filename}")
