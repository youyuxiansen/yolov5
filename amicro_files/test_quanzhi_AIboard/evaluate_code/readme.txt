1. 功能
检测文件夹中的jpg、png、bmp图片，并统一转为1280x720大小的nv21格式的YUV，并保存到yuv目录

2. 使用方法
（1）因为脚本需要调用ffmpeg，故先解压ffmpeg到脚本同一目录
（2）运行python脚本，选择要处理的图片目录，支持单目录和多目录处理（多目录请选择其上一级目录，会自动检测）

3. python库安装
pip install ffmpy3
pip install pillow
pip install threadsafe-tkinter
