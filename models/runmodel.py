import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import data
import numpy as np

def run_segment(img,net,size = 360):
    img = impro.resize(img,size)
    img = data.im2tensor(img, bgr2rgb = False, is0_1 = True)
    mask = net(img)
    mask = data.tensor2im(mask, gray=True, is0_1 = True)
    return mask

def run_pix2pixHD(img,net):
    img = impro.resize(img,512)
    img = data.im2tensor(img)
    img_fake = net(img)
    img_fake = data.tensor2im(img_fake)
    return img_fake

def get_mosaic_position(img_origin,net_mosaic_pos,mask_threshold=48,ex_mult=1.5,all_mosaic_area=False):
    h,w = img_origin.shape[:2]
    mask = run_segment(img_origin,net_mosaic_pos,size=360,)
    # mask_1 = mask.copy()
    mask = impro.mask_threshold(mask,ex_mun=int(min(h,w)/20),threshold=mask_threshold)
    if all_mosaic_area:
        mask = impro.find_mostlikely_ROI(mask)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=ex_mult)
    #Location fix
    rat = min(h,w)/360.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    x,y = np.clip(x, 0, w),np.clip(y, 0, h)
    size = np.clip(size, 0, min(w-x,h-y))
    # print(x,y,size)
    return x,y,size,mask