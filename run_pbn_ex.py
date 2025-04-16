from pbn import *

img_path = f'{os.getcwd()}/data/kayak.jpeg'
outpath = f'{os.getcwd()}/output/'
name = "Kayaking"
crayon_box = 96
no_crayon = True

create_pbn(img_path, name, outpath, crayon_box = crayon_box, pre_crayon_save = True)