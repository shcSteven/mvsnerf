from PIL import Image
import numpy as np
import os

def save_minmax(filename, min, max):
    file = open(filename, "wb")
    file.write("{} {}\n".format(min, max).encode('utf-8'))
    file.close()

for k in range(150):
    scan_path = os.path.join('./depths','scan%05d' % k)
    if(os.path.exists(scan_path)==False):
        continue
    
    for i in range(80):
        path = os.path.join('./depths','scan%05d' % k, '%03d.png' % i)
        depth_pil = Image.open(path)
        depth_image = np.array(depth_pil)
        pix = depth_image/1000.0
        min_value = 100
        max_value = pix[0,0]
        for x in range(512):
            for y in range(512):
                if(pix[x,y]>max_value):
                    max_value = pix[x,y]
                if(pix[x,y]<min_value and pix[x,y]!=0):
                    min_value = pix[x,y]

        minmax_file_path = os.path.join('./minmaxs', 'scan%05d' % k)
        os.makedirs(minmax_file_path, exist_ok=True)
        minmax_file_name = os.path.join(minmax_file_path,'minmax_map_%03d.txt' % i)
        save_minmax(minmax_file_name, min_value, max_value)
    