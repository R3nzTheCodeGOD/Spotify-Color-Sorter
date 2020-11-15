import numpy as np
import scipy.misc as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO
import urllib.request

class SpotifyColorSorter:
    def __init__(self, img, format='RGB', image_processing_size=None) -> None:
        if format == 'RGB':
            self.img = img
        elif format == 'BGR':
            self.img = self.img[..., ::-1]
        else:
            print("[COLOR SORTER ERROR] Format RGB veya BGR olmalıdır.")
            return
            

        if image_processing_size:
            img = Image.fromarray(self.img)
            self.img = np.asarray(img.resize(image_processing_size, Image.BILINEAR))

    def best_color(self, k=8, color_tol=10, plot=False) -> None:
        artwork = self.img.copy()
        self.img = self.img.reshape((self.img.shape[0]*self.img.shape[1], 3))

        clt = KMeans(n_clusters=k)
        clt.fit(self.img)
        hist = self.find_histogram(clt)
        centroids = clt.cluster_centers_

        colorfulness = [self.colorfulness(color[0], color[1], color[2]) for color in centroids]
        max_colorful = np.max(colorfulness)

        if max_colorful < color_tol:
            best_color = [230, 230, 230]
        else:
            best_color = centroids[np.argmax(colorfulness)]

        if plot:
            bar = np.zeros((50, 300, 3), dtype='uint8')
            square = np.zeros((50, 50, 3), dtype='uint8')
            start_x = 0

            for (percent, color) in zip(hist, centroids):
                end_x = start_x + (percent * 300)
                bar[:, int(start_x):int(end_x)] = color
                start_x = end_x
            square[:] = best_color

            plt.figure()
            plt.subplot(1, 3, 1)
            plt.title('Çizim')
            plt.axis('off')
            plt.imshow(artwork)

            plt.subplot(1, 3, 2)
            plt.title('Çıkarılan renk sayısı = {}'.format(k))
            plt.axis('off')
            plt.imshow(bar)

            plt.subplot(1, 3, 3)
            plt.title('Renk {}'.format(square[0][0]))
            plt.axis('off')
            plt.imshow(square)
            plt.tight_layout()

            plt.plot()
            plt.show(block=False)
            plt.waitforbuttonpress()

        return best_color[0], best_color[1], best_color[2]

    def find_histogram(self, clt) -> None:
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        hist, _ = np.histogram(clt.labels_, bins=num_labels)

        hist = hist.astype('float')
        hist /= hist.sum()

        return hist

    def colorfulness(self, r, g, b) -> None:
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        
        rg_mean, rg_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

    
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)


def get_artwork(url) -> None:
    try:
        url = url
    except IndexError:
        print("[IMAGE ERROR] Bu geçersiz bir URL")
    image_bytes = BytesIO(urllib.request.urlopen(url).read())
    image = np.array(Image.open(image_bytes))
    return image

if __name__ == "__main__":
    img = get_artwork("https://i.scdn.co/image/ab67616d0000b273b102238e0e636acbbd304bbb")
    
    image_color = SpotifyColorSorter(img, format='RGB', image_processing_size=None)
    image_color.best_color(plot=True)
    
    


    