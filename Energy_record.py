import xlrd
import xlwt
from xlutils.copy import copy
import numpy as np

#nvidia-smi dmon -i 0 -s p -d 5 -o TD  to record the power of the GPU
def write_excel(file_name, sheet_name, title):
    index = len(title)  # Get the number of rows of data to be written
    workbook = xlwt.Workbook()
    for name in sheet_name:
        sheet = workbook.add_sheet(name)  # Create a new table in the workbook
        for i in range(index):
            sheet.write(i, 0, title[i])  # Write data in the table (i = rows and j = columns)
    workbook.save(file_name)


def append_excel(file_name, data, sheet_num):
    index = len(data) 
    workbook = xlrd.open_workbook(file_name)
    sheets = workbook.sheet_names() 
    worksheet = workbook.sheet_by_name(sheets[sheet_num])
    cols_old = worksheet.ncols  # get the rows number
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(sheet_num)
    for i in range(index):
        for j in range(len(data[i])):
            new_worksheet.write(i+1, j+cols_old, data[i][j])  # add append data
    new_workbook.save(file_name)
    
def update_B(register, c, E_list):
    for i in range(len(register)):
        if i == c.client_id:
            E_list[i].append(register[c.client_id]['Drone']['Battery'])
            break
        else:
            continue
    return register, E_list

def update_U(conf, used_E, c, U_list):
    for i in range(conf['no_models']):
        if i == c.client_id:
            U_list[i].append(used_E)
            break
        else:
            continue
    return U_list