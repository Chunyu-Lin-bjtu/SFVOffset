'''
从初始记录文件名的txt中生成偏移文件名的txt
'''

import os

readFileName = "train_usefile_3dobj.txt"
writeFileName = "val.txt"

counter = 0 
with open(readFileName) as readFileProj:
    lines = readFileProj.readlines()
    with open(writeFileName, 'w') as writeFileProj:
        for line in lines:
            if counter % 3 ==1:
                lineStr = line.rstrip('\n')
                # idex, move_num = lineStr.split(' ')
                # str_idex = "%06d"%num_idex
                lineStr = lineStr.zfill(6)

                writeFileProj.write(lineStr + '\n')
                counter += 1
            else:
                counter += 1
                continue
            
        