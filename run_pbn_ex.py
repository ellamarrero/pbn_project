from pbn import *

img_path = f'{os.getcwd()}/data/kayak.jpeg' # image filepath, should be added to 'data' folder
outpath = f'{os.getcwd()}/output/' # output pathfile
name = "Kayaking" # name of image (will also be included in output files)
crayon_box = 96 # number of crayons in box -- (16, 24, 48, 64, 96, 120)
pre_crayon_save = True # whether or not you want a version of simplified image before colors are replaced with crayon colros

create_pbn(img_path, name, outpath, crayon_box = crayon_box, pre_crayon_save = pre_crayon_save)