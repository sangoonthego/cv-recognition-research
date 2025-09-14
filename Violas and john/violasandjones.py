import numpy as np
import cv2

def rect_sum(inter_image, x, y, w, h):
    A = inter_image[y, x]
    B = inter_image[y, x + w]
    C = inter_image[y + h, x]
    D = inter_image[y + h, x + w]
    return D - B - C + A

def haar2rec_horizone(x, y, inter_image, scale):
    H, W = inter_image.shape
    size_haar = np.array([2, 1]) * scale
    w, h = int(size_haar[0]), int(size_haar[1])
    if x + w > W or y + h > H:
        return None
    left = rect_sum(inter_image, x, y, w // 2, h)
    right = rect_sum(inter_image, x + w // 2, y, w // 2, h)
    return right - left

def haar2rec_vertical(x, y, inter_image, scale):
    H, W = inter_image.shape
    size_haar = np.array([1, 2]) * scale
    w, h = int(size_haar[0]), int(size_haar[1])
    if x + w > W or y + h > H:
        return None
    top = rect_sum(inter_image, x, y, w, h // 2)
    bottom = rect_sum(inter_image, x, y + h // 2, w, h // 2)
    return bottom - top

def haar3rec(x, y, inter_image, scale):
    H, W = inter_image.shape
    size_haar = np.array([3, 1]) * scale
    w, h = int(size_haar[0]), int(size_haar[1])
    if x + w > W or y + h > H:
        return None
    left = rect_sum(inter_image, x, y, w // 3, h)
    center = rect_sum(inter_image, x + w // 3, y, w // 3, h)
    right = rect_sum(inter_image, x + 2 * w // 3, y, w // 3, h)
    return center - left - right

def haar4rec(x, y, inter_image, scale):
    H, W = inter_image.shape
    size_haar = np.array([2, 2]) * scale
    w, h = int(size_haar[0]), int(size_haar[1])
    if x + w > W or y + h > H:
        return None
    tl = rect_sum(inter_image, x, y, w // 2, h // 2)
    tr = rect_sum(inter_image, x + w // 2, y, w // 2, h // 2)
    bl = rect_sum(inter_image, x, y + h // 2, w // 2, h // 2)
    br = rect_sum(inter_image, x + w // 2, y + h // 2, w // 2, h // 2)
    return tl - tr - bl + br


class ViolaJones:
    def __init__(self, haar_func, window_size, num_classifiers=50, scale=1):
        self.scale = scale
        self.window_size = window_size
        self.num_classifiers = num_classifiers
        self.haar_func = haar_func
        self.weak_classifiers = []
        self.features = None

    def generate_feature(self):
        feature = []
        H, W = self.window_size
        for i in range(H):
            for j in range(W):
                for haar_func in self.haar_func:
                    if haar_func == haar2rec_horizone:
                        w, h = 2 * self.scale, 1 * self.scale
                    elif haar_func == haar2rec_vertical:
                        w, h = 1 * self.scale, 2 * self.scale
                    elif haar_func == haar3rec:
                        w, h = 3 * self.scale, 1 * self.scale
                    elif haar_func == haar4rec:
                        w, h = 2 * self.scale, 2 * self.scale
                    else:
                        continue
                    w, h = int(w), int(h)
                    if j + w > W or i + h > H:
                        continue
                    feature.append([j, i, w, h, haar_func])
        self.features = feature
    def apply_feature(self, ii, feature):
        x, y, w, h, haar_func = feature
        return haar_func(x, y, ii, self.scale)

    @staticmethod
    def find_best(featureis, labels, w):
        sort_idx = np.argsort(featureis)
        featureis = featureis[sort_idx]
        labels = labels[sort_idx]
        w = w[sort_idx]

        total_pos = np.sum(w * labels)
        total_neg = np.sum(w * (1 - labels))
        S_neg, S_pos = 0, 0
        best_error = float('inf')
        best_thresh, best_pol = None, 1

        for i in range(len(featureis)):
            if labels[i] == 1:
                S_pos += w[i]
            else:
                S_neg += w[i]
            error = min(S_pos + (total_neg - S_neg),
                        S_neg + (total_pos - S_pos))
            if error < best_error:
                best_error = error
                best_thresh = featureis[i]
                if S_pos + (total_neg - S_neg) < S_neg + (total_pos - S_pos):
                    best_pol = 1
                else:
                    best_pol = -1

        return best_error, best_pol, best_thresh

    @staticmethod
    def week_pred(ii, clf):
        feature, pol, thresh = clf
        x, y, w, h, haar_func = feature
        f = haar_func(x, y, ii, 1)
        if f is None:
            return 0
        if pol == 1:
            return 1 if f < thresh else 0
        else:
            return 1 if f > thresh else 0

    def train(self, images,labels,images_vals,labels_vals): # bổ sung thêm val test
        integral_image = [cv2.integral(img) for img in images]
        n = len(images)
        w = np.ones(n) / n

        for t in range(self.num_classifiers):
            best_error = float('inf')
            best_clf = None

            for feature in self.features:
                featureis = np.array([self.apply_feature(ii, feature) for ii in integral_image])
                error, pol, thresh = self.find_best(featureis, labels, w)
                if error < best_error:
                    best_error = error
                    best_clf = [feature, pol, thresh]

            alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-10))
            pred = np.array([self.week_pred(ii, best_clf) for ii in integral_image])
            w = w * np.exp(-alpha * (2 * labels - 1) * (2 * pred - 1))
            w /= w.sum()

            self.weak_classifiers.append((best_clf, alpha))
            if images_vals is not None and labels_vals is not None:
              pred=[]
              for image in images_vals: 
                pred.append(self.predict(image))
              pred=np.array(pred)
              acc=(pred==labels_vals).mean()*100
              print(f"num_classifiers {t+1}: error={best_error:.4f} val_accuracy={acc:.2f}")
            else: 
              print(f"num_classifiers {t+1}: error={best_error:.4f}")

    def predict(self, image):
        ii = cv2.integral(image)
        H, alpha_sum = 0, 0
        for clf, alpha in self.weak_classifiers:
            pred = self.week_pred(ii, clf)
            H += alpha * pred
            alpha_sum += alpha
        return 1 if H >= 0.5 * alpha_sum else 0
