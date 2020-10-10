import sys
import urllib.request
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def spotify(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray_chkfinite(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def find_histogram(clt):
    numlabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numlabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50,300,3), dtype="uint8")
    startx = 0

    for (percent, color) in zip(hist, centroids):
        endx = startx + (percent * 300)
        cv2.rectangle(bar, (int(startx), 0), (int(endx), 50), color.astype("uint8").tolist(), -1)
        startx = endx
    return bar


if __name__ == "__main__":
    image = spotify("https://i.scdn.co/image/ab67616d0000b273e78ca25a967d525711df9cfe")
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    img = cv2.imread("Image", 0)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3))
    clt = KMeans(n_clusters=2)
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plt.show()


    