import pandas as pd
import os
from pathlib import Path

#ranges = pd.read_csv(os.path.join(currDir,"video_data","rangeTable.csv"))
#videoTable = pd.read_csv(os.path.join(currDir,"video_data","videoTable.csv"))

#videos = videoTable['Id'].unique()

def download_video(id):
    currDir = Path(__file__).parent
    sourceDir = os.path.join(currDir,"videos","source")
    if (not os.path.isdir(sourceDir)):
        os.makedirs(sourceDir)
    if (os.path.exists(os.path.join(sourceDir,id + ".mp4"))):
        return
    cmd = f'youtube-dl.exe -o "{sourceDir}/%(id)s.mp4" -f mp4 https://www.youtube.com/watch?v=' + id
    print("Executing: " + cmd)
    os.system(cmd)

def delete_video(id):
    currDir = Path(__file__).parent
    sourceDir = os.path.join(currDir,"videos","source")
    filePath = os.path.join(sourceDir,id + ".mp4")
    if (os.path.exists(filePath)):
        os.remove(filePath)