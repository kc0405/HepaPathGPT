import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

# 定义输入和输出目录
input_dir = '/mnt/newdisk/MultiLLMdata/data/chongyisan/data'  # 替换为你的nii文件所在的文件夹路径
output_dir = '/mnt/newdisk/MultiLLMdata/data/chongyisan/datajpg'  # 替换为jpg输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个患者的nrrd文件
for patient_id, nrrd_file in enumerate(sorted(os.listdir(input_dir)), start=1):
    # 确保是nrrd文件
    if nrrd_file.endswith('.nrrd'):
        # 加载nrrd文件
        nrrd_path = os.path.join(input_dir, nrrd_file)
        img_data, header = nrrd.read(nrrd_path)

        # 创建一个新文件夹为每个患者存储jpg
        patient_folder = os.path.join(output_dir, str(patient_id))
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

        # 遍历每个切片并保存为jpg
        for slice_idx in range(img_data.shape[2]):
            slice_data = img_data[:, :, slice_idx]
            slice_data = np.rot90(slice_data)  # 旋转图像，使其方向正确
            jpg_filename = os.path.join(patient_folder, f'slice_{slice_idx+1}.jpg')

            # 检查文件是否已存在，避免覆盖
            if not os.path.exists(jpg_filename):
                # 使用matplotlib保存为jpg
                plt.imsave(jpg_filename, slice_data, cmap='gray')
            else:
                print(f"文件 {jpg_filename} 已存在，跳过保存。")

        print(f"已保存患者 {patient_id} 的所有切片为 JPG。")
