import os
import cv2
import multiprocessing
import zipfile
import torch
import time
import cv2
import os
import numpy as np

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from PIL import Image


model = torch.hub.load("ultralytics/yolov5", "yolov5n")
model = torch.hub.load('./', 'custom', path='Breast_Cropper.pt', source='local')

#Take prediction from model from taken image as parameter / Gogus goruntusunun ROI kısmını almak üzere eğitilmiş model çekilir
def capture(image):
    try:
        detections = model(image)
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        frame = image[int(results[0]['ymin']):int(results[0]['ymax']),int(results[0]['xmin']):int(results[0]['xmax'])]
        return frame    
    except:
        return image
	
#Linear strecth algorithm for Dicom to PNG process
#Dicom görüntülerinin dönüştürülmesinde linear stretch yöntemi kullanıldı
def lin_stretch_img(img, low_prc, high_prc, do_ignore_minmax=True):
    
    if do_ignore_minmax:
        tmp_img = img.copy()
        med = np.median(img)  
        tmp_img[img == img.min()] = med
        tmp_img[img == img.max()] = med
    else:
        tmp_img = img

    lo, hi = np.percentile(tmp_img, (low_prc, high_prc))  

    if lo == hi:
        return np.full(img.shape, 128, np.uint8)  

    stretch_img = (img.astype(float) - lo) * (255/(hi-lo))  
    stretch_img = stretch_img.clip(0, 255).astype(np.uint8)  
    
    return stretch_img

#Extracting PNG from Dicom
def dicom_to_PNG(filename):

    ds = pydicom.read_file(filename) 
    img = ds.pixel_array 
    img = apply_voi_lut(img, ds, index=0)
    img = lin_stretch_img(img, 0.1, 99.9)  

    
    if ds[0x0028, 0x0004].value == 'MONOCHROME1':
        img = 255-img 
    
    return capture(img)

def extract_folders(zip_ref, dest_folder, zip_info_list):
    for zip_info in zip_info_list:
        if zip_info.is_dir():
            extracted_folder_path = os.path.join(dest_folder, zip_info.filename)
            os.makedirs(extracted_folder_path, exist_ok=True)

            subfolder_items = [zi for zi in zip_info_list if zi.filename.startswith(zip_info.filename + "/")]
            extract_folders(zip_ref, extracted_folder_path, subfolder_items)
        else:
            zip_ref.extract(zip_info, dest_folder)
            dcm_file = dest_folder + zip_info.filename
            name = zip_info.filename[:-4] + '.png'
            cv2.imwrite(dest_folder + name, dicom_to_PNG(dest_folder + zip_info.filename))
            os.remove(dcm_file)

def process_zip_file(zip_path, dest_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_folders(zip_ref, dest_folder, zip_ref.infolist())

if __name__ == '__main__':
    zip_path = 'D:/Open.zip'
    dest_folder = 'D:/Dest/'
    num_processes = multiprocessing.cpu_count()

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_info_list = zip_ref.infolist()
        chunks = [zip_info_list[i::num_processes] for i in range(num_processes)]
        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(process_zip_file, [(zip_path, dest_folder) for chunk in chunks])

        pool.close()
        pool.join()
