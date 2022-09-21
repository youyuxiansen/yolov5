import os
import shutil
import cv2
from tkinter.filedialog import (askopenfilename,askopenfilenames,askdirectory,asksaveasfilename)
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def Files(path):
    for pathes, dires, files in os.walk(path):
        return files

def Folders(path):
    for pathes, dires, files in os.walk(path):
        return dires

def word():
    pass

def pic_handle(dir,font):
    null_dir = dir + "/null"
    paint_dir = dir + "/paint"
    info = "recoginfo.txt"
    if(os.path.exists(dir + '/' +info)):
        print("Find recoginfo.txt")
    else:
        print("<Error> Can't find recoginfo.txt")
        return
    if(not os.path.exists(null_dir)):
        os.mkdir(null_dir)
    if(not os.path.exists(paint_dir)):
        os.mkdir(paint_dir)
    f = open(dir + '/' + info,"r")
    lines = f.readlines()
    f.close()
    zero = "0000"
    LabelDict = { 0:"数据线",1:"鞋",2:"垃圾桶",3:"毯子",4:"抹布",5:"粪便",6:"体重秤",7:"袜子",8:"污渍",9:"风扇座",10:"纸团"}
    for line in lines:
        line_tmp = line.split(' ')
        fname = line_tmp[0]
        src_image = dir + '/' + fname
        w = line_tmp[3]
        h = line_tmp[4]
        if((zero in w) and (zero in h)):
            shutil.copyfile(src_image,null_dir + '/' + fname)
        else:
            dst_image = paint_dir + '/' + fname
            xmin = int(line_tmp[1].split(':')[-1])
            ymin = int(line_tmp[2].split(':')[-1])
            xmax = int(xmin) + int(w.split(':')[-1])
            ymax = int(ymin) + int(h.split(':')[-1])
            label = int(line_tmp[-2].split(':')[-1])
            dist = int(line_tmp[-1].split(':')[-1])
            text = LabelDict[label]+" "+str(dist)+"cm"
            print(text)
            # print(xmin,ymin,xmax,ymax)
            image = cv2.imread(src_image)
            first_point = (xmin, ymin)
            last_point = (xmax, ymax)
            cv2.rectangle(image, first_point, last_point, (0, 0, 255), 3)
            cv2.imwrite(dst_image, image)
            tp=Image.open(dst_image)
            # 在图片上添加文字 1
            if(ymin > 30):
                drawx = xmin
                drawy = ymin - 40
            else:
                drawx = xmin
                drawy = ymax + 2
            draw = ImageDraw.Draw(tp)
            draw.text((drawx, drawy),text,(255,0,0),font=font)
            draw = ImageDraw.Draw(tp)
            tp.save(dst_image)
            # cv2.imwrite(dst_image, image)

if __name__ == "__main__":
    # directory = askdirectory()
    directory = "/home/yousixia/project/yolov5/amicro_files/test_quanzhi_AIboard/dataset_for_val/pic"
    folders = Folders(directory)
    files = Files(directory)
    font = ImageFont.truetype("amicro_files/test_quanzhi_AIboard/evaluate_code/simsun.ttc", 40)
    for fn in files:
        if "recoginfo.txt" in fn:
            pic_handle(directory,font)
            exit()

    for folder in folders:
        dir = directory + "/" + folder
        print("Find folder: %s"%(folder))
        pic_handle(dir,font)
