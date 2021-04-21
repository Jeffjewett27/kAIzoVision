from download_sources import *
from summarize_data import *
from create_folders import createImgDirs
from video_splitter import split_video
import sys

keepvideosources = (len(sys.argv) > 1) and bool(sys.argv[1])
print(keepvideosources)

#createTrainTestDirs()
createImgDirs()

ids = getJSONFileIds(True)
old_processed = readProcessedIds()
print(ids)
ranges = getRangeFrame(ids)
processed = []
def prepareVideo(id, frame):
    download_video(id)
    subset = frame[frame["Id"] == id]
    (ret,imgs) = split_video(id, frame)
    print(f"Video processed with image frame shape: {imgs.shape}")
    writeProcessedImgs(imgs)
    if (ret):
        writeProcessedIds([id])
        processed.append(id)
        appendRangeTable(subset)
        frame.drop(subset,inplace=True)
    if (not keepvideosources):
        delete_video(id)

for id in ids:
    if not (id in old_processed):
        prepareVideo(id, ranges)