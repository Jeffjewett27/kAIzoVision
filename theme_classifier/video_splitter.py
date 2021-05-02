import pandas as pd
from pathlib import Path
from glob import glob
import shlex
import os
import subprocess
import logging
from categories import *
from summarize_data import *

minClipLength = 5
maxClipLength = 20
skipFactor = 4

logging.basicConfig(filename='video_splitter.log', level=logging.DEBUG)

videodir = os.path.join(Path(__file__).parent, "videos")
imgdir = os.path.join(videodir, "images")
traindir = os.path.join(videodir, "train_videos")
testdir = os.path.join(videodir, "test_videos")

relvideodir = "./videos"
relimgdir = os.path.join(relvideodir, "images")

def rangeFilePath(row, category):
    category = category_filename(category)
    filename = row['Id'] + "_" + str(row['VideoIndex'])
    path = category + "/" + filename
    return path

def split_video(id, manifest):
    """ Split video into segments based on the given manifest table"""
    images = pd.DataFrame(columns=["index","Filename","Style","Theme","Time","Train"])

    sourceFile = os.path.join(videodir, "source", id + ".mp4")
    all_good = True
    for index, row in manifest.iterrows():
        print(row)
        split_str = ""
        split_args = []
        try:
            split_start = row["Start"]
            split_end = row["End"]
            split_length = split_end - split_start
            if (split_length < minClipLength):
                continue
            skip_step = max((skipFactor * split_length) // maxClipLength, 1)
            extra = f'-s 512x512 -vf "framestep=step={skip_step}"'

            category = decode_category(row["Category"])
            filepath = rangeFilePath(row,category)
            glb = glob(filepath + '*.jpg')
            if (len(glb) > 0):
                print("skipping id: " + id + ", " + str(row["VideoIndex"]) + " at " + filepath)
                continue
            
            split_cmd = ["ffmpeg", "-i", sourceFile] + shlex.split(extra)
            split_args += ["-ss", str(split_start), "-t",
                str(split_length), os.path.join(imgdir, filepath + "_%04d.jpg")]
            print("########################################################")
            print("About to run : "+" ".join(split_cmd+split_args))
            print("########################################################")
            subprocess.check_output(split_cmd+split_args)

            glb = glob(os.path.join(imgdir, filepath + '*.jpg'))
            images = images.append(pd.DataFrame({
                "Filename": glb[len(imgdir)+1:],
                "Style": category[0],
                "Theme": category[1],
                "Time": category[2],
                "Train": row["Train"]
            }).reset_index())
        except Exception as e:
            logging.error("New Error")
            logging.error(row)
            logging.error(e)
            #all_good = False

    return (all_good, images)

def regenerateImageTable():
    ranges = getRangeFrame(getJSONFileIds(False))
    imgs = pd.DataFrame(columns=[
        "Filename",
        "Style",
        "Theme",
        "Time",
        "Train"
    ])
    for index, row in ranges.iterrows():
        category = decode_category(row['Category'])
        filepath = rangeFilePath(row,category)
        glb = glob(os.path.join(relimgdir, filepath + '*.jpg'))
        imgs = imgs.append(pd.DataFrame({
            "Filename": [x[len(imgdir)+1:].replace("\\","/") for x in glb],
            "Style": category[0],
            "Theme": category[1],
            "Time": category[2],
            "Train": row["Train"]
        }), ignore_index=True)
    imgs = imgs.reset_index()
    writeProcessedImgs(imgs,True)

if __name__ == "__main__":
    regenerateImageTable()