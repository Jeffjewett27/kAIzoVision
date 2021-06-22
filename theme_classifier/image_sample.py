import numpy as np

def tableStyleWeights(ranges):
    #weights = ranges[ranges['Train']][["Style","Length"]].groupby("Style").sum()["Length"]
    weights = ranges["Style"].value_counts()
    return (weights / weights.mean()).to_dict()

def tableThemeWeights(ranges):
    #weights = ranges[ranges['Train']][["Theme","Length"]].groupby("Theme").sum()["Length"]
    weights = ranges[ranges['Train']]["Theme"].value_counts()
    return (weights / weights.mean()).to_dict()

def tableTimeWeights(ranges):
    #weights = ranges[ranges['Train']][["Time","Length"]].groupby("Time").sum()["Length"]
    weights = ranges[ranges['Train']]["Time"].value_counts()
    return (weights / weights.mean()).to_dict()

def row_weight_function(ranges):
    style_w = tableStyleWeights(ranges)
    theme_w = tableThemeWeights(ranges)
    time_w = tableTimeWeights(ranges)
    theme_w[''] = 1
    time_w[''] = 1
    print(style_w)
    print(theme_w)
    print(time_w)
    return lambda x: 1 if x.Train != True else style_w.get(x.Style) * theme_w.get(x.Theme) * time_w.get(x.Time)

def sample_images(images):
    images = images.replace(np.nan, '', regex=True)
    images["Weight"] = images.apply(row_weight_function(images[images.Train]), axis = 1)

    '''under = images[images.Weight > 1]
    print("under:",under.shape[0])
    under_w = 1 - 1 / under.Weight
    under = under.sample(weights = under_w)
    print("under:",under, under_w)

    over = images[images.Weight <= 1]
    over = over.append(over.sample(weights = 1 - over.Weight))

    under.append(over).to_csv("video_data/sampledImages.csv")'''
    samp = images.sample(frac=1, replace=True, weights= 1/images.Weight)
    samp = samp.drop(["index"], axis=1).reset_index(drop=True).reset_index()
    samp.to_csv("video_data/sampledImages.csv", index=False)