import SimpleITK as sitk
import os

def tif2nii(data_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    file_names = [os.path.join(data_path,f) for f in file_list]
    
    newspacing = [1, 1, 1]  # 设置x，y, z方向的空间间隔
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    vol.SetSpacing(newspacing)
    sitk.WriteImage(vol, 'volume.nii') # 保存为volume.nii.gz也可