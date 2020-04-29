# Author : Daulet Baimukashev
# Convert rosbag to image + labels dataset

import csv, sys
import numpy as np
from PIL import Image
import re, cv2
import time, os

start_time = time.time()
csv.field_size_limit(sys.maxsize)

filename = "_slash_all_data_calib_normal_shear_torsion.csv"

# Loop through remaining lines in file object f
with open(filename, "rb") as csvfile:
  datareader = csv.reader(csvfile)
  count = 0
  icount = 0
  for line in datareader:
    img = np.zeros((480,640,3))
    data_img = np.zeros(480*3*640)
    data_imgEdited = np.zeros(480*3*640)
    icount = 0
    if count < 1:
        print(line)
    if count > 0: #and count < 2:     # "1528778092065871962" in line: #count==2:
      #print(line[31])
      var1_operation_mode = line[7]
      var2_current_state = line[8]
      var3_pointX = line[10]
      var3_pointY = line[11]
      var3_pointZ = line[12]
      var4_angleZ = line[13 ]
      var5_fx = line[14]
      var5_fy = line[15]
      var5_fz = line[16]
      var5_fwx = line[17]
      var5_fwy = line[18]
      var5_fwz = line[19]
      var6_camera = line[32]

      data_img = line[32].split(',')
      # print(len(data_img))
      for i in range(len(data_img)-1):
          data_imgEdited[i] = (re.search(r'\d+', data_img[i]).group())
      #data_imgEdited = list(map(int, data_imgEdited))
      #print(data_imgEdited)
      #print(len(data_imgEdited))
      #print(line[1:20])
      #print('featute',line[31])

      #array = np.linspace(0,1,256*256)
      # reshape to 2d
      img = np.reshape(data_imgEdited,(480,640,3)) # BGR
      #print(img)
      #print('shape', img.shape)
      # Creates PIL image
      img = Image.fromarray(np.uint8(img) , 'RGB')
      #imagename = "images/image" + str(count) + ".png"
      #img.save(imagename)

      b, g, r = img.split()
      img2 = Image.merge("RGB", (r, g, b))

      #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      imagename = "images/imagexx" + str(count) + ".png"
      img2.save(imagename)

      #line18 = line[18].replace("[","")
      #line18 = line18.replace("]","")
      #line18 = line18.replace(" ","")
      #line18 = line18.replace(" ","")
      #line18 = map(float, line18.split(','))
      #print(line[18])d
      #print(line18)

      with open('train_labels.csv', "a") as csvfile:
        csvwriter = csv.writer(csvfile,  delimiter=',')
        #line_to_write = list()
        #line_to_write.append([count,line[7],line[8],line[9],line[10], line[11],line[12],line[13]])
        #line_to_write.extend(line[18])
        #csvwriter.writerow(line_to_write)
        csvwriter.writerow([count, var1_operation_mode, var2_current_state,var3_pointX,var3_pointY,var3_pointZ, var4_angleZ, var5_fx, var5_fy,var5_fz, var5_fwx,var5_fwy,var5_fwz])
    #with open('somefile.txt', 'a') as the_file:
	#a = []
	#word = str(count) + str(line[2])
	#a = ",".join(word)
	##a.append(count)
	#print(a)
	#the_file.write(word)

      #img.show()
      #print(line[26])
      #data_img = line[26]
      #for i in range(0,480):
	#for j in range(0,640):
	  #icount = icount + 1
	  #img[i,j] = data_img[icount]
      #img = line[26]
      ##fields = line.split(",")
    #if count > 105 :
    #  break
    count = count + 1

print("--- %s seconds ---" % (time.time() - start_time))

#os.remove("_slash_all_data_calib.csv")
