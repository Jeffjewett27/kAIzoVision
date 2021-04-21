import os
import sys
from categories import *
from pathlib import Path


#print("create dirs at " + sys.argv[1])
def createTrainTestDirs():
    videodir = os.path.join(Path(__file__).parent, "videos")
    traindir = os.path.join(videodir, "train_videos")
    testdir = os.path.join(videodir, "test_videos")
    createFolders(traindir)
    createFolders(testdir)

def createImgDirs():
    videodir = os.path.join(Path(__file__).parent, "videos")
    imagedir = os.path.join(videodir, "images")
    createFolders(imagedir)

def createFolders(dir):
    for cat in list_decoded():
        folder = category_filename(cat)
        newpath = os.path.join(dir,folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)