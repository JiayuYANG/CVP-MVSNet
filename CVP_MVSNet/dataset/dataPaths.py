# Fetching file path and name for dataloader on our DTU dataset. 
# by: Jiayu Yang
# date: 2019-08-01

import os

# DTU:
# 2020-01-31 14:20:42: Modified to read original yao's format.
def getScanListFile(data_root,mode):
    scan_list_file = data_root+"scan_list_"+mode+".txt"
    return scan_list_file

def getPairListFile(data_root,mode):
    pair_list_file = data_root+"Cameras/pair.txt"
    return pair_list_file

def getDepthFile(data_root,mode,scan,view):
    depth_name = "depth_map_"+str(view).zfill(4)+".pfm"
    if mode == "train":
        scan_path = "Depths/"+scan+"_train/"
    else:
        scan_path = "Depths/"+scan+"/"
    depth_file = os.path.join(data_root,scan_path,depth_name)
    return depth_file

def getImageFile(data_root,mode,scan,view,light):
    image_name = "rect_"+str(view+1).zfill(3)+"_"+str(light)+"_r5000.png"
    if mode == "train":
        scan_path = "Rectified/"+scan+"_train/"
    else:
        scan_path = "Rectified/"+scan+"/"
    image_file = os.path.join(data_root,scan_path,image_name)
    return image_file

def getCameraFile(data_root,mode,view):
    cam_name = str(view).zfill(8)+"_cam.txt"
    cam_path = "Cameras/"
    cam_file = os.path.join(data_root,cam_path,cam_name)
    return cam_file

