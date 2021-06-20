import os 

filename = 'train_remove.txt'
write_filename = 'train.txt'
with open(filename, 'r') as file_txt_obj:
    with open(write_filename, 'w') as write_file_obj:
        i_counter = 0
        for line in file_txt_obj.readlines():
            if i_counter % 3 != 0:
                i_counter += 1
                continue
            line = line.strip('\n')
            line_str = line.split(' ')
            filename_write = line_str[0]
            remove_num = float(line_str[1])
            # print(filename_write, remove_num)

            remove_num = round(remove_num*4)*0.25
            write_file_obj.write(filename_write + '\n')
            if remove_num == 0:
                write_file_obj.write(filename_write + '\n')
                write_file_obj.write(filename_write + '\n')
            elif remove_num == 0.25:
                lineStr_Rl = filename_write + "removeYl0_25" + "\n"
                lineStr_Rr = filename_write + "removeYr0_25" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 0.5:
                lineStr_Rl = filename_write + "removeYl0_5" + "\n"
                lineStr_Rr = filename_write + "removeYr0_5" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 0.75:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 1.0:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 1.25:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 1.5:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 1.75:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            elif remove_num == 2.0:
                lineStr_Rl = filename_write + "removeYl0_75" + "\n"
                lineStr_Rr = filename_write + "removeYr0_75" + "\n"
                write_file_obj.write(lineStr_Rl)
                write_file_obj.write(lineStr_Rr)
            else:
                write_file_obj.write(filename_write + '\n')
                write_file_obj.write(filename_write + '\n')
            i_counter += 1
        
