import os, json
import pandas as pd
import numpy as np
from categories import *
from image_sample import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from video_splitter import minClipLength

def extendFrameCategories(frame):
    tup = frame['Category'].map(lambda x: decode_category(x))
    frame['Style'] = tup.map(lambda x: x[0])
    frame['Theme'] = tup.map(lambda x: x[1])
    frame['Time'] = tup.map(lambda x: x[2])

"""def setTrainColumn(frame, dayPct, nightPct):
    frame['Train'] = False

    nDay = frame[frame.Time == ""].shape[0]
    nNight = frame[frame.Time == "night"].shape[0]
    print(nDay)
    print(nNight)
    dayPerm = np.random.permutation(nDay)[:int(np.floor(dayPct*nDay)-1)]
    nightPerm = np.random.permutation(nNight)[:int(np.floor(nightPct*nNight)-1)]

    trainDayIndex = frame[frame.Time == ""].index[dayPerm].values.tolist()
    trainNightIndex = frame[frame.Time == "night"].index[nightPerm].values.tolist()

    frame.loc[trainDayIndex, "Train"] = True
    frame.loc[trainNightIndex, "Train"] = True"""

def setTrainColumn(frame, test_size):
    if (len(frame.axes[0]) == 0):
        frame['Train'] = True
        return frame
    train, test = train_test_split(frame, test_size=test_size,random_state=27)
    train.loc[:,'Train'] = True
    test.loc[:,'Train'] = False
    return train.append(test)

#path_to_json = os.path.join(__file__, "..")
#json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#print(json_files)

def readRangeIds():
    table_path = os.path.join(Path(__file__).parent, 'video_data', 'rangeTable.csv')
    if (not os.path.exists(table_path)):
        return set()
    videos = pd.read_csv(table_path)
    return set(videos['Id'].unique())

def readProcessedIds():
    table_path = os.path.join(Path(__file__).parent, 'video_data', 'processedTable.csv')
    if (not os.path.exists(table_path)):
        return set()
    videos = pd.read_csv(table_path)
    return set(videos['Id'].unique())

def writeProcessedIds(ids):
    new_videos = pd.Series(ids)

    table_path = os.path.join(Path(__file__).parent, 'video_data', 'processedTable.csv')
    if (os.path.exists(table_path)):
        videos = pd.read_csv(table_path)['Id']
        videos = pd.DataFrame(videos.append(new_videos).unique(), columns=['Id'])
    else:
        videos = pd.DataFrame(new_videos, columns=['Id'])
    videos.to_csv(table_path, index=False)

def readProcessedImgs():
    table_path = os.path.join(Path(__file__).parent, 'video_data', 'imageTable.csv')
    if (os.path.exists(table_path)):
        return pd.read_csv(table_path)
    return pd.DataFrame()
    
def writeProcessedImgs(imgs, overwrite=False):
    table_path = os.path.join(Path(__file__).parent, 'video_data', 'imageTable.csv')
    if (not overwrite and os.path.exists(table_path)):
        images = pd.read_csv(table_path)
        images = images.append(imgs)
        print(images.head())
        #images.drop("index",columns=1,inplace=True)
        #images.reset_index(inplace=True)
        images.to_csv(table_path, index=False)
    else:
        imgs.to_csv(table_path,index=False)

def getJSONFileIds(onlyNew=False):
    json_dir = os.path.join(Path(__file__).parent, 'video_data')
    ids = {pos_json[:-5] for pos_json in os.listdir(json_dir) if pos_json.endswith('.json')}
    if (onlyNew):
        ids = ids.difference(readProcessedIds())
    return ids

def getRangeFrame(ids):
    json_dir = os.path.join(Path(__file__).parent, 'video_data')
    ranges = pd.DataFrame(columns=['Start','End','Category','Id','VideoIndex','Channel','Style','Theme','Time','Length'])

    for id in ids:
        fname = id + ".json"
        with open(os.path.join(json_dir, fname),'r') as file:
            data = file.read()
            obj = json.loads(data)
            frame = pd.DataFrame(obj.get("ranges"))
            frame['Id'] = obj.get('id')
            frame['Channel'] = obj.get('channel')
            frame['Length'] = frame['End'] - frame['Start']
            frame.reset_index(inplace=True)
            frame.rename(columns={'index':'VideoIndex'}, inplace=True)
            extendFrameCategories(frame)

            ranges = ranges.append(frame, ignore_index=True)

    ranges = setTrainColumn(ranges, 0.2)
    
    ranges.loc[:,"Weight"] = ranges.apply(row_weight_function(ranges[ranges.Train]), axis=1)
    return ranges

def appendRangeTable(frame):
    table_path = os.path.join(Path(__file__).parent, 'video_data', 'rangeTable.csv')
    ranges = frame
    if (os.path.exists(table_path)):
        ranges = pd.read_csv(table_path).append(frame, ignore_index=True)
    ranges.to_csv(table_path, index=False)


#print (ranges.head(20))
#print (ranges.describe())

#ranges.to_csv(os.path.join(path_to_json,"rangeTable.csv"))

#print(ranges.drop_duplicates(subset=["Category"]).sort_values(by=["Style","Theme","Time"]).head(6))