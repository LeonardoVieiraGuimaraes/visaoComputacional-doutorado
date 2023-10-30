# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:03:02 2021

@author: leona
"""

import os
import shutil

# dir = "C:/Users/leona/Desktop/Doutorado/VisaoComputacional/Quiz2"

files = os.listdir("./Fotos")


for cont,file in enumerate(files):
    
    shutil.copyfile("./Fotos/" + file,"./Fotos/imagem" + str(cont) + ".jpg")
    