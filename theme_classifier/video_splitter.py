import pandas as pd
from pathlib import Path
from glob import glob
import shlex
import os
import subprocess
import logging
from categories import *
from summarize_data import *

minClipLength = 2
maxClipLength = 20
skipFactor = 4
shortThresh = 10

logging.basicConfig(filename='video_splitter_debug.log', level=logging.DEBUG)
logging.basicConfig(filename='video_splitter_info.log', level=logging.INFO)
logging.basicConfig(filename='video_splitter_error.log', level=logging.ERROR)

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
    good_count = 0
    for index, row in manifest.iterrows():
        print(row)
        split_str = ""
        split_args = []
        try:
            split_start = row["Start"]
            length = row["Length"]
            if (length < minClipLength):
                logging.info("skipping id: " + id + ", " + str(row["VideoIndex"]) + " because it is too short: " + str(length))
                continue
            if (length > shortThresh):
                #trim a second off both ends, since this often has wrong labels
                split_start += 1
                length -= 1
            skip_step = int(max((skipFactor * row["Weight"] * length / maxClipLength, 1)))

            extra = f'-s 512x512 -vf "framestep=step={skip_step}"'

            category = decode_category(row["Category"])
            filepath = rangeFilePath(row,category)
            glb = glob(filepath + '*.jpg')
            if (len(glb) > 0):
                logging.info("skipping id: " + id + ", " + str(row["VideoIndex"]) + " at " + filepath)
                continue
            
            split_cmd = ["ffmpeg", "-i", sourceFile] + shlex.split(extra)
            split_args += ["-ss", str(split_start), "-t",
                str(length), os.path.join(imgdir, filepath + "_%04d.jpg")]
            print("########################################################")
            print("About to run : "+" ".join(split_cmd+split_args))
            print("########################################################")
            subprocess.check_output(split_cmd+split_args)

            glb = glob(os.path.join(imgdir, filepath + '*.jpg'))
            images = images.append(pd.DataFrame({
                "Filename": [os.path.relpath(x,imgdir) for x in glb],
                "Style": category[0],
                "Theme": category[1],
                "Time": category[2],
                "Train": row["Train"]
            }).reset_index())
            logging.info(f"Completed {row['Style']}, {row['Theme']}, {row['Time']}, {row['Train']} in {id}, {row['VideoIndex']} with {len(glb)} frames")
            good_count += 1
        except Exception as e:
            logging.error("New Error")
            logging.error(row)
            logging.error(e)
            print(e)
            #all_good = False
    logging.info(f"Successfully processed {good_count} ranges for video {id}")
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
        #print(os.path.relpath(glb[0],imgdir))
        imgs = imgs.append(pd.DataFrame({
            "Filename": [os.path.relpath(x,imgdir) for x in glb],
            "Style": category[0],
            "Theme": category[1],
            "Time": category[2],
            "Train": row["Train"]
        }), ignore_index=True)
    imgs = imgs.reset_index()
    writeProcessedImgs(imgs,True)

if __name__ == "__main__":
    regenerateImageTable()