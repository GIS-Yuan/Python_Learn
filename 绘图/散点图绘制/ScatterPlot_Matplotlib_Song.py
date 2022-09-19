# import

import os
import gdal
import numpy as np
from numpy import polyfit,poly1d
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import log

import math

from sklearn.metrics import r2_score

# 忽略在取对数过程中，0取对数的Runtimewarning
import warnings
warnings.filterwarnings("ignore")
# 获取图像的行列数、投影信息、栅格数据
def readtif(filepath):
    dataset = gdal.Open(filepath)
    # 栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 栅格矩阵的行数
    im_height = dataset.RasterYSize
    # 获取影像投影转换信息
    im_geotrans = dataset.GetGeoTransform()
    # 获取地图投影（空间参照系统）信息
    im_proj = dataset.GetProjection()
    # 获取栅格数据（数组形式）
    im_data = dataset.ReadAsArray()
    del dataset
    # [0]列数 [1]行数 [2]投影转换信息 [3]地图投影信息 [4]栅格数据
    return [im_width, im_height, im_geotrans, im_proj, im_data]

def BatchDraw(xfilepath, yfilepath, outputfilepath):
    files = os.listdir(xfilepath)
    tifs = [tif for tif in files if tif[-3:] == 'tif']
    for tif in tifs:
        # 读取 TIFF 文件并提取 NoData 填充值
        extension = readtif(os.path.join(xfilepath, tif))
        viirs = readtif(os.path.join(yfilepath, tif))
        efillvalue = np.min(extension[4])
        vfillvalue = np.min(viirs[4])
        # 像元值和填充值取对数
        logextension = np.log(extension[4])  # 绘图阶段不用
        logviirs = np.log(viirs[4])
        logefillvalue = np.log(efillvalue)   # 绘图阶段不用
        logvfillvalue = np.log(vfillvalue)
        
        # 二维数组展平
        extension_array = extension[4].flatten()                 # 取对数 1/4
        viirs_array = logviirs.flatten()                         # 取对数 2/4
        df = pd.DataFrame()
        df["Extension"] = extension_array
        df["VIIRS"] = viirs_array
        df = df.drop(df[df["Extension"] == efillvalue].index)    # 取对数 3/4
        df = df.drop(df[df["VIIRS"] == logvfillvalue].index)     # 取对数 4/4
        
        df = df.drop(df[df["VIIRS"] < 0.2].index)
        df_reindex = df.reset_index()
        
        output = os.path.join(outputfilepath, tif[:-3] + 'png')

        # 绘制散点图
        plt.scatter(df["VIIRS"], df["Extension"], c="lightblue",alpha=0.5, marker=".")
        plt.xlim(xmax=5.2,xmin=0)
        
        # 添加x、y轴的文字说明
        plt.xlabel("SNPP-VIIRS")
        plt.ylabel("Extending-DMSP")
        plt.savefig(output, dpi = 500)
        
        # plt.show()
        # 每次画图完成后清空当前的 axis，避免出现 RuntimeWarning: More than 20 figures have been opened.
        # 使用 plt.figure() 与 plt.close() 也可以，但是 plt.cla() 避免重复创建与删除操作，复用一个 figure
        plt.cla()
        print(tif, 'is processed.')
    print('All done.')
    
if __name__ == '__main__': 
    dnl_dir = r"F:\Data\Extension_Research\2 Cropping\edmsp\cncity\1sh"
    vnl_dir = r"F:\Data\Extension_Research\2 Cropping\resamv\cncity\1sh"
    output_dir = r"F:\Data\Extension_Research\5 LogScatterPlot\viirs\cncity\1sh"

    BatchGetFigures(dnl_dir, vnl_dir, output_dir)
    print("completed.")
