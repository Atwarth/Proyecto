import os
import shutil

dest = r"C:\Users\Jorge Eliecer\Desktop\server\static\logs\logs.txt"

f = open(dest,'w')

for i in range(10):
    
    for j in range(100):
        f.write(f"{i+j}")
        f.write("\n")


#f.close()