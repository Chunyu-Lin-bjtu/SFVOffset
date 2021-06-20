# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:41:58 2018

@author: Administrator
"""
#!/usr/bin/python

import os

def num2txt(dir, file, num): # 写入文件数量限制
  for i in range(num):
    temp_str = '%010d' % i
    file.write(temp_str + '\n')

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    files.sort() # 对文件名进行排序
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion): # 如果文件类型中有文件夹，则继续扫描文件夹中的文件
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(os.path.splitext(name)[0] + "\n") # 不将文件后缀名写入
                    # file.write(os.path.splitext(name)[0]+ os.path.splitext(name)[1] + "\n") # for test.txt
                    break

def Test():
  dir="./lidar_2d/"     #文件路径
  outfile="./allnpy.txt"                     #写入的txt文件名
  # wildcard = ".jpg .exe .dll .lib .bmp .png"      #要读取的文件类型；
  wildcard = ".npy"

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)
  # num2txt(dir, file, 390)

  file.close()


Test()