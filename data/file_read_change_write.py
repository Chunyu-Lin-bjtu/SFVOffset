""" 针对每个图片的最佳偏移距离生成偏移的训练数据
"""
import os

readFileName = "train_usefile_3dobj.txt"
writeFileName = "train.txt"

counter = 0 
with open(readFileName) as readFileProj:
    lines = readFileProj.readlines()
    with open(writeFileName, 'w') as writeFileProj:
        for line in lines:
            if counter % 3 ==0:
                lineStr = line.rstrip('\n')
                idex, move_num = lineStr.split(' ')
                # str_idex = "%06d"%num_idex
                lineStr = idex.zfill(6)
                if float(move_num) == 0:
                    lineStr_Rl1 = lineStr + "\n"
                    lineStr_Rr1 = lineStr + "\n"
                else:
                    num_integer, num_decimal = move_num.split('.')
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
            
        