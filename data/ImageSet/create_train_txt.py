'''
从初始记录文件名的txt中生成偏移文件名的txt
'''

import os

readFileName = "train_usefile_3dobj.txt"
writeFileName = "train.txt"

counter = 0 
move_num = 0.5
with open(readFileName) as readFileProj:
    lines = readFileProj.readlines()
    with open(writeFileName, 'w') as writeFileProj:
        for line in lines:
            if counter % 3 ==0:
                lineStr = line.rstrip('\n')
                # idex, move_num = lineStr.split(' ')
                # str_idex = "%06d"%num_idex
                lineStr = lineStr.zfill(6)
                if move_num == 0.0:
                    lineStr_Rl1 = lineStr + "\n"
                    lineStr_Rr1 = lineStr + "\n"
                else:
                    move_num_str = str(move_num)
                    num_integer, num_decimal = move_num_str.split('.')
                    lineStr_Rl1 = lineStr + "removeYl" + num_integer + "_" + num_decimal + "\n"
                    # lineStr_Rr1 = lineStr + "removeYr" + num_integer + "_" + num_decimal + "\n"
                    lineStr_Rr1 = lineStr + "removeYl" + num_integer + "_" + num_decimal + "\n"

                writeFileProj.write(lineStr + '\n')
                writeFileProj.write(lineStr_Rl1)
                writeFileProj.write(lineStr_Rr1)
                counter += 1
            else:
                counter += 1
                continue
            
        