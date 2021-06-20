'''
用于rawdata数据集
'''
import os

readFileName = "train_original.txt"
writeFileName = "train_new.txt"

counter = 0 
with open(readFileName) as readFileProj:
    lines = readFileProj.readlines()
    with open(writeFileName, 'w') as writeFileProj:
        for line in lines:
            """
            if counter % 3 ==0:
                lineStr = line.rstrip('\n')
                lineStr_Rl1 = lineStr + "l0_5" + "\n"
                lineStr_Rr1 = lineStr + "removeYl0_1" + "\n"
                # lineStr_Rl1 = lineStr + "\n"
                # lineStr_Rr1 = lineStr + "\n"
                writeFileProj.write(lineStr + '\n')
                writeFileProj.write(lineStr_Rl1)
                writeFileProj.write(lineStr_Rr1)
                counter += 1
            else:
                counter += 1
                continue
            """
            lineStr = line.rstrip('\n')
            lineStr_Rl1 = lineStr + "l0_5" + "\n"
            lineStr_Rr1 = lineStr + "r0_5" + "\n"
            writeFileProj.write(lineStr + '\n')
            writeFileProj.write(lineStr_Rl1)
            writeFileProj.write(lineStr_Rr1)

            