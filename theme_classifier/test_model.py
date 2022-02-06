from model import get_trained_model, get_multilabelbinarizer, table_path, imagedir, mlb_inverse_transform_batch, get_one_hots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


def display_classification_sample(model, imgs, n):
    print("displaying " + str(n) + " imgs")
    mlb = get_multilabelbinarizer()
    samp = imgs.sample(n=n).reset_index().drop("level_0",axis=1)
    print(samp)
    img_list = []
    for i in range(n):
        fname = os.path.join(imagedir, samp.loc[i,"Filename"])
        img = plt.imread(fname)
        img_list.append(img)
    img_list.append(np.zeros((512,512,3)))
    imgs = np.asarray(img_list)
    outputs = model.predict(imgs)
    print(outputs)

    one_hots = get_one_hots()
    print(one_hots[0].categories)
    classes = mlb_inverse_transform_batch(outputs, one_hots)
    print(classes)

    for i in range(n):

        #multihot = model.predict(imgin,verbose=1)[0]
        #print("multi",multihot)
        print(classes[i])

        plt.imshow(img_list[i])
        plt.title(classes[i])
        plt.show()

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("must pass arguments: n weights")
        sys.exit(1)
    n = int(sys.argv[1])
    weights = sys.argv[2]
    model = get_trained_model(False, weights)
    imgs = pd.read_csv(table_path)
    display_classification_sample(model, imgs, n)

