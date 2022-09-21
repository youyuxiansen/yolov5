from ffmpy3 import FFmpeg
import os
from PIL import Image
from tkinter.filedialog import (askopenfilename,askopenfilenames,askdirectory,asksaveasfilename)
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm


def Files(path):
    for pathes, dires, files in os.walk(path):
        return files

def Folders(path):
    for pathes, dires, files in os.walk(path):
        return dires

def pic2yuv_nv21(dir,fname):
    pic_path = dir + "/" + fname
    yuv_dir = dir + "/yuv"
    if(not os.path.exists(yuv_dir)):
        os.mkdir(yuv_dir)
    pic = fname.split(".")[0]
    out_wid = 1280
    out_hei = 720
    size = '{}x{}'.format(out_wid,out_hei)
    outname = yuv_dir + '/' + pic + '_'+size+ '.yuv'
    ff = FFmpeg(executable="amicro_files/test_quanzhi_AIboard/evaluate_code/ffmpeg",inputs={pic_path:None},
    outputs={outname:'-s {} -pix_fmt nv21'.format(size)})
    # print(ff.cmd)
    ff.run()

if __name__ == "__main__":
    # directory = askdirectory()
    directory = "/data/yousixia/images/shujutang_data/shujutang_for_quanzhi_val"
    folders = Folders(directory)
    files = Files(directory)
    # font = ImageFont.truetype("./simsun.ttc",40)
    for fn in tqdm(files):
        if ".jpg" in fn or ".png" in fn or ".bmp" in fn:
            pic2yuv_nv21(directory,fn)

    for folder in folders:
        dir = directory + "/" + folder
        files = Files(dir)
        for fn in tqdm(files):
            if ".jpg" in fn or ".png" in fn or ".bmp" in fn:
                pic2yuv_nv21(dir,fn)
