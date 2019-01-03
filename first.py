import xlrd,pandas

ti = xlrd.open_workbook("Titles.xlsx")

print(ti.nsheets)
print(ti.sheet_names())

t11 = ti.sheet_by_index(0)

print(t11.row_values(0))
print(t11.col_values(0))
print(t11.cell(0,0))
print(t11.cell(0,0).value)
print(t11.row_slice(rowx=0,start_colx=0,end_colx=2))

