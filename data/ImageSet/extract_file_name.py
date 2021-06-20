# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:41:58 2018

@author: Administrator
"""
#!/usr/bin/python

import os

def num2txt(dir, file, num):
  for i in range(num):
    temp_str = '%010d' % i
    file.write(temp_str + '\n')

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(os.path.splitext(name)[0] + "\n")
                    # file.write(os.path.splitext(name)[0]+ os.path.splitext(name)[1] + "\n") # for test.txt
                    break

def Test():
  dir="./data/"     #文件路径
  outfile="./all.txt"                     #写入的txt文件名
  # wildcard = ".jpg .exe .dll .lib .bmp .png"      #要读取的文件类型；
  wildcard = ".png"

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  # ListFilesToTxt(dir,file,wildcard, 1)
  num2txt(dir, file, 390)

  file.close()


Test()