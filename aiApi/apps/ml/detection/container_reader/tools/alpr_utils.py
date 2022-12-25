from cv2 import kmeans
# from yolov4.tf import YOLOv4
import cv2
import numpy as np
import math
import tensorflow.compat.v1 as tf
# from skimage import morphology
# import skimage
# from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans
# from skimage.filters import *
# import pytesseract as tess
import tools.yolov4_utils as yolo_utils

from PIL import ImageFont, ImageDraw, Image

yolo_confidence = 0.2


def load_model_yolo(class_path: str, weights_path: str, input_size: int = 256):
    print("Yolo Loading...")
    yolo = YOLOv4(isTest=True, input_size=input_size)
    yolo.classes = class_path
    print(f"class is {yolo.classes}")
    yolo.make_model()
    yolo.load_weights(weights_path, weights_type="yolo")
    return yolo


def auto_write_balance(gray_img, mean_val=90):
    # cal mean of raw_img_gray image.
    # gray_mean = np.mean(gray_img)
    mean = np.mean(gray_img)
    print(f"auto_write_balance mean: {mean}")
    if mean < mean_val:
        wb_img = cv2.equalizeHist(gray_img)
    else:
        wb_img = gray_img
    return wb_img


def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def histeq(img, mean_val=90):
    # print(f"img len: {len(img.shape)}")
    # print(f"mean_val: {mean_val}")
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsvSplit = cv2.split(hsv)

        mean = np.mean(hsvSplit[2])
        median = np.median(hsvSplit[2])

        # print(f"mean: {mean}")
        # print(f"median: {median}")
        if mean < mean_val:
            hsvSplit[2] = cv2.equalizeHist(hsvSplit[2])

        hsv = cv2.merge(hsvSplit)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif len(img.shape) == 2:
        mean = np.mean(img)
        median = np.median(img)
        # print(f"mean: {mean}")
        # print(f"median: {median}")
        if mean < mean_val:
            out = cv2.equalizeHist(img)
        else:
            out = img

    return out


def yolo_crop_img(img, img_gray, candidates, prop=0.5):
    img_crop = None
    if len(candidates) > 0:
        img_crop = []
        w = img.shape[1]
        h = img.shape[0]
        # Set propability
        if candidates.shape[-1] == 5:
            candidates = np.concatenate(
                [candidates, np.full((*candidates.shape[:-1], 1), 2.0)], axis=-1
            )
        else:
            candidates = np.copy(candidates)

        # Convert ratio to length
        candidates[:, [0, 2]] = candidates[:, [0, 2]] * w
        candidates[:, [1, 3]] = candidates[:, [1, 3]] * h
        # print(candidates)
        # Draw bboxes
        for i, bbox in enumerate(candidates):
            center_x, center_y, cw, ch, class_id, propability = bbox
            if propability > prop:
                img_c = img.copy()
                img_c_gray = img_gray.copy()

                c_x = int(bbox[0])
                c_y = int(bbox[1])
                half_w = int(bbox[2] / 2)
                half_h = int(bbox[3] / 2)

                # print(c_x, c_y, half_w, half_h)
                x1 = int(c_x - half_w)
                x2 = int(c_x + half_w)

                y1 = int(c_y - half_h)
                y2 = int(c_y + half_h)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h
                # print(x1, x2, y1, y2)
                crop_img = img_c[int(y1):int(y2), int(x1):int(x2), :]
                crop_img_gray = img_c_gray[int(y1):int(y2), int(x1):int(x2)]
                img_crop.append([crop_img, center_x, center_y, cw, ch, class_id, propability,
                                 plate_identify(crop_img_gray)])
    return img_crop


def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def cal_slope(p1, p2):
    # print(f"cal_slope: {p1},{p2}")
    if p2[0] - p1[0] == 0:
        return 0.00000000001
    else:
        return float(p2[1] - p1[1]) / float(p2[0] - p1[0])


def RotatePoint(p: list, rad: float):
    x = math.cos(rad) * p[0] - math.sin(rad) * p[1]
    y = math.sin(rad) * p[0] + math.cos(rad) * p[1]
    return [x, y]


def distanceBetweenPoints(p1, p2):
    asquared = float((p2[0] - p1[0]) * (p2[0] - p1[0]))
    bsquared = float((p2[1] - p1[1]) * (p2[1] - p1[1]))
    # print(f"asquared: {asquared}")
    # print(f"bsquared: {bsquared}")
    return float(math.sqrt(asquared + bsquared))


def angleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    return math.atan2(float(deltaY), float(deltaX)) * (180.0 / np.pi)


def plate_identify(img_gray):
    # dn = cv2.bilateralFilter(img_gray, 11, 17, 17)
    # dn = cv2.GaussianBlur(img_gray, (5, 5), 2)
    # _, bin_img = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # img_mask = cv2.bitwise_and(img_gray, bin_img)

    # dn = cv2.fastNlMeansDenoising(img_gray, 3, 7, 21)
    dn = cv2.GaussianBlur(img_gray, (5, 5), 2)
    dn = cv2.bilateralFilter(dn, 11, 17, 17)
    # mask = cv2.threshold(Denoising, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask_src = cv2.adaptiveThreshold(dn, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    HistImg = cv2.bitwise_and(img_gray, mask_src)

    Denoising = cv2.bilateralFilter(HistImg, 11, 17, 17)
    bin_img = cv2.adaptiveThreshold(Denoising, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 30)
    # _,bin_img = cv2.threshold(bin_img, 60, 255, cv2.THRESH_BINARY_INV)

    kernel0 = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel0)

    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_HITMISS, kernel)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel3)

    # img_mask = cv2.bitwise_and(img_gray, bin_img)

    img_draw = img_gray.copy()
    got_text = False
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count_char = 0
    for c in sort_contours(contours):
        x, y, w, h = cv2.boundingRect(c)
        ratio = h / w
        print(f"ratio: {ratio}")
        if 1.0 <= ratio <= 7.0:  # Only select contour with defined ratio
            print(f"h / img_gray.shape[0]: {h / img_gray.shape[0]}")
            if 0.17 <= h / img_gray.shape[0] < 0.5:
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count_char = count_char + 1
            if count_char > 2:
                got_text = True
                break

    # cv2.imshow("HistImg",HistImg)
    # cv2.imshow("bin_img",bin_img)
    # cv2.imshow("img_draw",img_draw)
    # cv2.waitKey(0)
    return got_text

def get_licence_plate(plate_detector, raw_img_color, img_gray):
    # predict plate boundary.
    results = plate_detector.predict(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB))
    print(results)
    cv2.imshow("draw_bboxes", plate_detector.draw_bboxes(raw_img_color, results))
    # results
    return yolo_crop_img(raw_img_color.copy(), img_gray, results, 0.7)


def pad(img, h, w, gray_value=255, mode='constant'):
    h = h + img.shape[0]
    w = w + img.shape[1]
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    if mode == 'constant':
        return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode=mode,
                              constant_values=gray_value))
    elif mode == 'reflect':
        return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode=mode,
                              reflect_type='even'))

    else:
        return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode=mode))


class PlateLine:
    def __init__(self, line=None, confidence=0.0):
        self.line = line
        self.confidence = confidence
    # def __init__(self,line,confidence):
    #    self.line = line
    #    self.confidence = confidence


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.p1 = (int(x1), int(y1))
        self.p2 = (int(x2), int(y2))
        self.slope = cal_slope(self.p1, self.p2)
        self.length = distanceBetweenPoints(self.p1, self.p2)
        self.angle = angleBetweenPoints(self.p1, self.p2)

    def getPointAt(self, x: float):
        return float(self.slope * (x - float(self.p2[0])) + float(self.p2[1]))

    def getXPointAt(self, y: float):
        y_intercept = self.getPointAt(0)
        return float((y - y_intercept) / self.slope)

    def getParallelLine(self, distance: float):
        diff_x = self.p2[0] - self.p1[0]
        diff_y = self.p2[1] - self.p1[1]
        angle = math.atan2(diff_x, diff_y)
        dist_x = distance * math.cos(angle)
        dist_y = -distance * math.sin(angle)

        offsetX = int(round(dist_x))
        offsetY = int(round(dist_y))

        return LineSegment(self.p1[0] + offsetX, self.p1[1] + offsetY, self.p2[0] + offsetX, self.p2[1] + offsetY)

    def closestPointOnSegmentTo(self, p):
        top = (p[0] - self.p1[0]) * (self.p2[0] - self.p1[0]) + (p[1] - self.p1[1]) * (self.p2[1] - self.p1[1])
        bottom = distanceBetweenPoints(self.p2, self.p1)
        bottom = bottom * bottom
        u = float(top) / float(bottom)
        x = self.p1[0] + u * (self.p2[0] - self.p1[0])
        y = self.p1[1] + u * (self.p2[1] - self.p1[1])
        return [int(x), int(y)]

    def intersection(self, line):
        intersection_X = -1
        intersection_Y = -1
        c1 = self.p1[1] - self.slope * self.p1[0]
        c2 = line.p2[1] - line.slope * line.p2[0]
        if (self.slope - line.slope) == 0:
            print("No Intersection between the lines")
        elif self.p1[0] == self.p2[0]:
            return [int(self.p1[0]), int(line.getPointAt(self.p1[0]))]
        elif line.p1[0] == line.p2[0]:
            return [int(line.p1[0]), int(self.getPointAt(line.p1[0]))]
        else:
            intersection_X = (c2 - c1) / (self.slope - line.slope)
            intersection_Y = self.slope * intersection_X + c1
        return [int(intersection_X), int(intersection_Y)]


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w <= 0 or h <= 0: return (0, 0, 0, 0)  # or (0,0,0,0) ?
    return (x, y, w, h)


def area(rect):
    return rect[2] * rect[3]


def shape_area(img):
    return img.shape[0] * img.shape[1]


def get_8mark(mark_model, img,
              offset_x: np.ndarray = np.array([0, 0, 0, 0]),
              offset_y: np.ndarray = np.array([0, 0, 0, 0]),
              dest_size=256,
              channel=3,
              reshape=True,
              plate_denoise=None,
              DEBUG=False
              ):
    offset_x = offset_x / 255.
    offset_y = offset_y / 255.
    if DEBUG:
        print(f"offset_x: {offset_x}")
        print(f"offset_y: {offset_y}")
        print(f'img shape: {img.shape}')
        print(f'dest_size: {dest_size}')
    if reshape:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # add blur
        # img_gray_smoothed = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # cv2.bilateralFilter()
        # img_gray_smoothed = img_gray.copy()
        # img_gray_smoothed = cv2.bilateralFilter(img_gray, 5, 75, 75)
        img_gray_smoothed = cv2.medianBlur(img_gray, 3)
        _, _, plate_denoise = resize_for_implement_keras(img_gray_smoothed, desired_size=dest_size)
        # cv2.imshow("plate_denoise", plate_denoise)

        if channel == 3:
            plate_denoise = cv2.cvtColor(plate_denoise, cv2.COLOR_GRAY2RGB)

        plate_denoise = plate_denoise / 255.
        plate_denoise = plate_denoise.astype(np.float32)
        plate_denoise = plate_denoise.reshape(-1, dest_size, dest_size, channel)  # return each images as 96 x 96 x 3
    _points = None
    _points = mark_model.predict(plate_denoise)

    if _points is None:
        return None
    if reshape:
        del img_gray
        del plate_denoise
        del img_gray_smoothed

    points = _points[0]
    if DEBUG:
        print(f"raw points: {points}")
    tl = [points[0] + offset_x[0], points[1] + offset_y[0]]
    tr = [points[2] + offset_x[1], points[3] + offset_y[1]]
    bl = [points[4] + offset_x[2], points[5] + offset_y[2]]
    br = [points[6] + offset_x[3], points[7] + offset_y[3]]

    if tl[0] < 0:
        tl[0] = 0
    if tl[1] < 0:
        tl[1] = 0

    if tr[0] < 0:
        tr[0] = 0
    if tr[1] < 0:
        tr[1] = 0

    if bl[0] < 0:
        bl[0] = 0
    if bl[1] < 0:
        bl[1] = 0

    if br[0] < 0:
        br[0] = 0
    if br[1] < 0:
        br[1] = 0
    points_raw = [tl[0], tl[1], tr[0], tr[1], bl[0], bl[1], br[0], br[1]]
    del points
    del _points
    if DEBUG:
        print(f"adjust points: {points_raw}")
    return points_raw


def draw_mark(_point4, img, dest_size=256):
    point4 = np.reshape(_point4.copy(), -1) * float(dest_size)
    # print(f"reshaped point4: {point4}")
    dtl = (int(point4[0]), int(point4[1]))
    dtr = (int(point4[2]), int(point4[3]))
    dbl = (int(point4[4]), int(point4[5]))
    dbr = (int(point4[6]), int(point4[7]))

    x_ratio, y_ratio, draw_circle = resize_for_implement_keras(img, dest_size)
    cv2.circle(draw_circle, dtl, 3, (255, 0, 0), 3)
    cv2.circle(draw_circle, dtr, 3, (0, 255, 0), 3)
    cv2.circle(draw_circle, dbl, 3, (0, 0, 255), 3)
    cv2.circle(draw_circle, dbr, 3, (255, 255, 0), 3)
    return draw_circle


def warp_image_v2(_point4, img, dest_size=256, DEBUG=False):
    point4 = np.reshape(_point4.copy(), -1) * float(dest_size)
    if DEBUG:
        print(f"reshaped point4: {point4}")
    dtl = (int(point4[0]), int(point4[1]))
    dtr = (int(point4[2]), int(point4[3]))
    dbl = (int(point4[4]), int(point4[5]))
    dbr = (int(point4[6]), int(point4[7]))
    pts1 = np.float32([[dtl[0], dtl[1]], [dtr[0], dtr[1]], [dbl[0], dbl[1]], [dbr[0], dbr[1]]])

    w1 = round(distanceBetweenPoints(dtl, dtr))
    h1 = round(distanceBetweenPoints(dtl, dbl))

    pts2 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
    # pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(f'M: {M}')
    _, _, warp = resize_for_implement_keras(img, dest_size, color=[0, 0, 0])
    warp = cv2.warpPerspective(warp, M, (w1, h1), flags=cv2.INTER_NEAREST)
    # warp = k_resize(warp, dest_size)
    return warp


def warp_image(_point4, img, dest_size=256, DEBUG=False):
    point4 = np.reshape(_point4.copy(), -1) * float(dest_size)
    if DEBUG:
        print(f"reshaped point4: {point4}")
    dtl = (int(point4[0]), int(point4[1]))
    dtr = (int(point4[2]), int(point4[3]))
    dbl = (int(point4[4]), int(point4[5]))
    dbr = (int(point4[6]), int(point4[7]))
    pts1 = np.float32([[dtl[0], dtl[1]], [dtr[0], dtr[1]], [dbl[0], dbl[1]], [dbr[0], dbr[1]]])

    # w1_slope = cal_slope(dtl, dtr)
    w1_distance = distanceBetweenPoints(dtl, dtr)
    w1_angle = angleBetweenPoints(dtl, dtr)
    w1_distance = w1_distance * math.cos(w1_angle)

    # h1_slope = cal_slope(dtl, dbl)
    h1_distance = distanceBetweenPoints(dtl, dbl)
    h1_angle = angleBetweenPoints(dtl, dbl)
    h1_distance = h1_distance * math.cos(h1_angle)

    if DEBUG:
        print(f"w1_distance: {w1_distance}")
        print(f"h1_distance: {h1_distance}")
    # w1 = int(w1_distance)
    # h1 = int(h1_distance)

    w1 = round(distanceBetweenPoints(dtl, dtr))
    h1 = round(distanceBetweenPoints(dtl, dbl))

    # w1 = dtr[0]-dtl[0]
    # h1 = dbl[1]-dtl[1]

    pts2 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
    # pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    _, _, warp = resize_for_implement_keras(img)
    warp = cv2.warpPerspective(warp, M, (w1, h1))
    return warp


def resize_image(img, size=(28, 28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.full((dif, dif), fill_value=255, dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.full((dif, dif, c), fill_value=255, dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def mish(x):
    return tf.keras.layers.Lambda(lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x))))(x)


# warp_resize = (W,H)
def get_char_mask(tfnet_chararea, warp, warp_resize: np.ndarray = None,
                  dest_size=(256, 256),
                  erode_iterations: int = -1,
                  erode_kernel=None,
                  dilate_iterations=None,
                  dilate_kernel=None,
                  DEBUG=False):
    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    warp_ratio_x = dest_size[0] / float(warp.shape[1])
    warp_ratio_y = dest_size[1] / float(warp.shape[0])
    offset = 5
    area_results = tfnet_chararea.return_predict(img=warp, index=[0, 1])
    if DEBUG:
        print(f'area_results: {area_results}')
    area_results = reconstruct_charbox(area_results)
    text_area_mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
    for area_result in area_results:
        if area_result['confidence'] > yolo_confidence:
            mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
            # point1 = (int(area_result['topleft']['x']),
            #           int(area_result['topleft']['y']))
            # point2 = (int(area_result['bottomright']['x']),
            #           int(area_result['bottomright']['y']))
            # cv2.rectangle(warp_gray, point1, point2, (255, 255, 255), -1)
            point1 = (int(area_result['topleft']['x'] * warp_ratio_x),
                      int((area_result['topleft']['y'] + offset) * warp_ratio_y))
            point2 = (int(area_result['bottomright']['x'] * warp_ratio_x),
                      int((area_result['bottomright']['y'] - offset) * warp_ratio_y))
            # cv2.rectangle(warp_rgb, point1, point2, (0, 255, 0), 1)
            cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
            if erode_iterations != -1:
                if erode_kernel is not None:
                    mask = cv2.erode(mask, erode_kernel, iterations=erode_iterations)
                else:
                    mask = cv2.erode(mask, None, iterations=erode_iterations)
            if dilate_iterations is not None:
                if dilate_kernel is not None:
                    mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)
                else:
                    mask = cv2.dilate(mask, None, iterations=dilate_iterations)
            text_area_mask = cv2.bitwise_or(text_area_mask, mask)

    text_area_mask = cv2.cvtColor(text_area_mask, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("warp_gray", warp_gray)
    return text_area_mask


# warp_resize = (W,H)
def get_text_mask(area_results, warp, warp_resize: np.ndarray = None,
                  dest_size=(256, 256),
                  erode_iterations: int = -1,
                  erode_kernel=None,
                  dilate_iterations=None,
                  dilate_kernel=None,
                  DEBUG=False):
    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    warp_ratio_x = dest_size[0] / float(warp.shape[1])
    warp_ratio_y = dest_size[1] / float(warp.shape[0])
    offset = 5
    # area_results = tfnet_chararea.return_predict(img=warp, index=[0, 1])
    if DEBUG:
        print(f'area_results: {area_results}')
    # area_results = reconstruct_charbox(area_results)
    text_area_mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
    for area_result in area_results:
        if area_result['confidence'] > yolo_confidence:
            mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
            # point1 = (int(area_result['topleft']['x']),
            #           int(area_result['topleft']['y']))
            # point2 = (int(area_result['bottomright']['x']),
            #           int(area_result['bottomright']['y']))
            # cv2.rectangle(warp_gray, point1, point2, (255, 255, 255), -1)
            point1 = (int(area_result['topleft']['x'] * warp_ratio_x),
                      int((area_result['topleft']['y'] + offset) * warp_ratio_y))
            point2 = (int(area_result['bottomright']['x'] * warp_ratio_x),
                      int((area_result['bottomright']['y'] - offset) * warp_ratio_y))
            # cv2.rectangle(warp_rgb, point1, point2, (0, 255, 0), 1)
            cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
            if erode_iterations != -1:
                if erode_kernel is not None:
                    mask = cv2.erode(mask, erode_kernel, iterations=erode_iterations)
                else:
                    mask = cv2.erode(mask, None, iterations=erode_iterations)
            if dilate_iterations is not None:
                if dilate_kernel is not None:
                    mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)
                else:
                    mask = cv2.dilate(mask, None, iterations=dilate_iterations)
            text_area_mask = cv2.bitwise_or(text_area_mask, mask)

    text_area_mask = cv2.cvtColor(text_area_mask, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("warp_gray", warp_gray)
    return text_area_mask


def get_text_maskWH(area_results, warp, warp_resize: np.ndarray = None,
                    dest_size=(256, 256),
                    erode_iterations: int = -1,
                    erode_kernel=None,
                    erode_iterations1: int = -1,
                    erode_kernel1=None,
                    dilate_iterations=None,
                    dilate_kernel=None,
                    DEBUG=False):
    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    warp_ratio_x = dest_size[0] / float(warp.shape[1])
    warp_ratio_y = dest_size[1] / float(warp.shape[0])
    offset = 5
    # area_results = tfnet_chararea.return_predict(img=warp, index=[0, 1])
    if DEBUG:
        print(f'area_results: {area_results}')
    # area_results = reconstruct_charbox(area_results)
    text_area_mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
    for area_result in area_results:
        if area_result['confidence'] > yolo_confidence:
            mask = np.zeros((dest_size[1], dest_size[0], 1), dtype=np.uint8)
            # point1 = (int(area_result['topleft']['x']),
            #           int(area_result['topleft']['y']))
            # point2 = (int(area_result['bottomright']['x']),
            #           int(area_result['bottomright']['y']))
            # cv2.rectangle(warp_gray, point1, point2, (255, 255, 255), -1)
            point1 = (int(area_result['topleft']['x'] * warp_ratio_x),
                      int((area_result['topleft']['y'] + offset) * warp_ratio_y))
            point2 = (int(area_result['bottomright']['x'] * warp_ratio_x),
                      int((area_result['bottomright']['y'] - offset) * warp_ratio_y))
            # cv2.rectangle(warp_rgb, point1, point2, (0, 255, 0), 1)
            cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
            if erode_iterations != -1:
                if erode_kernel is not None:
                    mask = cv2.erode(mask, erode_kernel, iterations=erode_iterations)
                else:
                    mask = cv2.erode(mask, None, iterations=erode_iterations)
            if erode_iterations1 != -1:
                if erode_kernel1 is not None:
                    mask = cv2.erode(mask, erode_kernel1, iterations=erode_iterations1)
                else:
                    mask = cv2.erode(mask, None, iterations=erode_iterations1)
            if dilate_iterations is not None:
                if dilate_kernel is not None:
                    mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)
                else:
                    mask = cv2.dilate(mask, None, iterations=dilate_iterations)
            text_area_mask = cv2.bitwise_or(text_area_mask, mask)

    text_area_mask = cv2.cvtColor(text_area_mask, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("warp_gray", warp_gray)
    return text_area_mask


def remove_mid_lines(warp):
    h, w, ch = warp.shape
    warp_rgb = warp.copy()
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    cv2.imshow("warp_rgb", warp_rgb)
    cv2.imshow("warp_gray", warp_gray)
    top = LineSegment(0, 0, w, 0)
    bottom = LineSegment(0, h, w, h)

    warp_rgb_lines = warp_rgb.copy()
    herls = getline(warp, warp_gray, False, 105)
    for herl in herls:
        distance_top = distanceBetweenPoints(herl.line.p1, top.p1)
        distance_bottom = distanceBetweenPoints(herl.line.p1, bottom.p1)
        if distance_top > 30 and distance_bottom > 30:  # or distance_top < 10 or distance_bottom < 10
            cv2.line(warp_rgb_lines, herl.line.p1, herl.line.p2, (255, 255, 255), 10, cv2.LINE_AA)
    return warp_rgb_lines


def resize_for_keras_model(img, desired_size: int = 256, color=[20, 20, 20], count_pic_char=0, channel=1
                           , BORDER=cv2.BORDER_CONSTANT):
    plate_denoise = img.copy()
    x_ratio = 0
    y_ratio = 0
    if 1:
        if channel == 3:
            plate_denoise = cv2.cvtColor(plate_denoise, cv2.COLOR_GRAY2RGB)
        old_size = plate_denoise.shape[:2]
        # print(f"old_size: {old_size}")
        h, w = plate_denoise.shape[:2]
        ratio = float(desired_size) / float(max(old_size))
        new_size = tuple([int(round(x * ratio)) for x in old_size])
        # print(f"new_size: {new_size}")
        x_ratio = old_size[0] / new_size[0]
        y_ratio = old_size[1] / new_size[1]

        plate_denoise = cv2.resize(plate_denoise, dsize=(new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top_v, bottom_v = delta_h // 2, delta_h - (delta_h // 2)
        left_v, right_v = delta_w // 2, delta_w - (delta_w // 2)

        # color = [20, 20, 20]
        plate_denoise = cv2.copyMakeBorder(plate_denoise, top_v, bottom_v, left_v, right_v, BORDER,
                                           value=color)
        plate_denoise = cv2.medianBlur(plate_denoise, 3)
        plate_denoise = cv2.dilate(plate_denoise, None, iterations=1)
    else:
        plate_denoise = resize_image(plate_denoise, (desired_size, desired_size))
        plate_denoise = cv2.medianBlur(plate_denoise, 3)
    # cv2.imshow("plate_denoise_" + str(count_pic_char), plate_denoise)
    plate_denoise = plate_denoise / 255.
    plate_denoise = plate_denoise.astype(np.float32)

    plate_denoise = plate_denoise.reshape(-1, desired_size, desired_size, channel)  # return each images as 96 x 96 x 1
    # print(f"plate_denoise shape: {plate_denoise.shape}")
    return [x_ratio, y_ratio, plate_denoise]


def resize_for_keras_model_rec(img, desired_size=[256, 256], color=[20, 20, 20], count_pic_char=0, channel=1,
                               DEBUG=False):
    plate_denoise = img.copy()
    x_ratio = 0
    y_ratio = 0
    if 1:
        old_size = plate_denoise.shape[:2]
        ratio = float(desired_size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        x_ratio = old_size[0] / new_size[0]
        y_ratio = old_size[1] / new_size[1]
        plate_denoise = cv2.resize(plate_denoise, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
        h, w = plate_denoise.shape[:2]

        delta_w = desired_size[0] - new_size[1]
        delta_h = desired_size[1] - new_size[0]
        if delta_w < 0:
            delta_w = 0
        if delta_h < 0:
            delta_h = 0
        top_v, bottom_v = delta_h // 2, delta_h - (delta_h // 2)
        left_v, right_v = delta_w // 2, delta_w - (delta_w // 2)

        # color = [0, 0, 0]
        plate_denoise = cv2.copyMakeBorder(plate_denoise, top_v, bottom_v, left_v, right_v, cv2.BORDER_CONSTANT,
                                           value=color)
        plate_denoise = cv2.medianBlur(plate_denoise, 3)
        # plate_denoise = cv2.er(plate_denoise, None, iterations=1)
        if DEBUG:
            cv2.imshow("plate_denoise", plate_denoise)
    else:
        plate_denoise = resize_image(plate_denoise, (desired_size, desired_size))
        plate_denoise = cv2.medianBlur(plate_denoise, 3)
    # cv2.imshow("plate_denoise_" + str(count_pic_char), plate_denoise)
    plate_denoise = plate_denoise.astype(np.float32)
    plate_denoise = plate_denoise / 255.
    plate_denoise = plate_denoise.reshape(-1, desired_size[1], desired_size[0],
                                          channel)  # return each images as 96 x 96 x 1
    # print(f"plate_denoise shape: {plate_denoise.shape}")
    return [x_ratio, (desired_size[1] / old_size[1]), plate_denoise]


def resize_for_implement_keras_rec(img, desired_size=[256, 256], color: list() = [20, 20, 20]):
    plate_denoise = img.copy()
    old_size = plate_denoise.shape[:2]
    ratio = float(desired_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    x_ratio = old_size[0] / new_size[0]
    y_ratio = old_size[1] / new_size[1]
    plate_denoise = cv2.resize(plate_denoise, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    delta_w = desired_size[0] - new_size[1]
    delta_h = desired_size[1] - new_size[0]
    if delta_w < 0:
        delta_w = 0
    if delta_h < 0:
        delta_h = 0
    top_v, bottom_v = delta_h // 2, delta_h - (delta_h // 2)
    left_v, right_v = delta_w // 2, delta_w - (delta_w // 2)

    # color = [0, 0, 0]
    plate_denoise = cv2.copyMakeBorder(plate_denoise, top_v, bottom_v, left_v, right_v, cv2.BORDER_CONSTANT,
                                       value=color)
    return [x_ratio, y_ratio, plate_denoise]


def resize_for_implement_keras(img, desired_size: int = 256, color: list() = [20, 20, 20], BORDER=cv2.BORDER_CONSTANT):
    plate_denoise = img.copy()
    old_size = plate_denoise.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    x_ratio = old_size[0] / new_size[0]
    y_ratio = old_size[1] / new_size[1]
    plate_denoise = cv2.resize(plate_denoise, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top_v, bottom_v = delta_h // 2, delta_h - (delta_h // 2)
    left_v, right_v = delta_w // 2, delta_w - (delta_w // 2)

    # color = [0, 0, 0]
    plate_denoise = cv2.copyMakeBorder(plate_denoise, top_v, bottom_v, left_v, right_v, BORDER,
                                       value=color)
    return [x_ratio, y_ratio, plate_denoise]


import zipfile
import platform
import os

# if platform.system() == 'Linux':
#     os.makedirs("/tmp/Chrome", exist_ok=True)
# elif platform.system() == 'Windows':
#     os.makedirs("C://ProgramData/Chrome", exist_ok=True)


# ** to update custom Activate functions
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add

""" Define layers block functions """


def Hswish(x):
    # print(x * tf.nn.relu6(x + 3) / 6)
    return x * tf.nn.relu6(x + 3) / 6


def relu6(x):
    """Relu 6
    """
    # print(x)
    return K.relu(x, max_value=6.0)


from tools.model import yolov3


class TF_model:
    def __init__(self, classes: list, new_size: list, anchors: list, restore_path, letterbox_resize: bool = False):
        self.num_class = len(classes)
        self.classes = classes
        self.anchors = anchors
        self.new_size = new_size
        self.restore_path = restore_path
        self.graph = tf.Graph()
        self.boxes = None
        self.scores = None
        self.labels = None
        self.letterbox_resize = letterbox_resize
        print(f'restore_path: {restore_path}')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=config)
            # with self.session as sess:
            self.input_data = tf.placeholder(tf.float32, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
            yolo_model = yolov3(self.num_class, self.anchors, use_label_smooth=True)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            self.boxes, self.scores, self.labels = self.gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200,
                                                                score_thresh=0.3,
                                                                nms_thresh=0.45)
            # nms_thresh=0.213)

            saver = tf.train.Saver()
            saver.restore(self.session, restore_path)

    def return_predict(self, img, channel=3, index: list = None,
                       reconstruct: bool = False,
                       threshold: float = 0.7,
                       input_size: list = [256, 256],
                       equalizeHist: bool = False,
                       _letterbox_resize=False,
                       DEBUG=False):

        boxes_ = None
        scores_ = None
        labels_ = None
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height_ori, width_ori = img_raw.shape[:2]

        if self.letterbox_resize or _letterbox_resize:
            img_raw, resize_ratio, dw, dh = self.letterbox_resize1(img_raw, self.new_size[1], self.new_size[0])
        else:
            img_raw = cv2.resize(img_raw, (self.new_size[1], self.new_size[0]), interpolation=cv2.INTER_LINEAR)
        img_raw = img_raw.astype(np.float32)
        img_raw = img_raw[np.newaxis, :] / 255.
        # cv2.imshow('img', img_raw[0])

        with self.graph.as_default():
            with self.session.as_default():
                boxes_, scores_, labels_ = self.session.run([self.boxes, self.scores, self.labels],
                                                            feed_dict={self.input_data: img_raw})

            if self.letterbox_resize or _letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(self.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(self.new_size[1]))
            # cv2.imshow("darkflow", self.draw_box(img_raw, res))
        # return boxes_, scores_, labels_
        return self.darkflow_format(boxes_, scores_, labels_, self.classes, threshold, index=index)

    def draw_box(self, img_line, results, color=(255, 0, 0), thickness=2, confidence: float = 0.5):
        if results is not None:
            for result in results:
                if result['confidence'] > confidence:
                    point1 = (result['topleft']['x'], result['topleft']['y'])
                    point2 = (result['bottomright']['x'], result['bottomright']['y'])
                    cv2.rectangle(img_line, point1, point2, color, thickness=thickness)
        return img_line

    def darkflow_format(self, boxes_, scores_, labels_, classes, threshold=0.5, index=None):
        boxesInfo = list()
        for i in range(len(boxes_)):
            coor = np.array(boxes_[i], dtype=np.float32)
            score = scores_[i]
            class_ind = int(labels_[i])
            if score > threshold and abs(coor[0] - coor[2]) > 0 and abs(coor[1] - coor[3]) > 0:
                x1 = coor[0]
                y1 = coor[1]
                x2 = coor[2]
                y2 = coor[3]

                if x1 < 0:
                    x1 = 0
                if x2 < 0:
                    x2 = 0
                if y1 < 0:
                    y1 = 0
                if y2 < 0:
                    y2 = 0

                # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                c1, c2 = (x1, y1), (x2, y2)
                # convert(image.shape[:2], bbox)
                if index is not None:
                    for idx in index:
                        if class_ind == idx:
                            boxesInfo.append({
                                "label": classes[class_ind],
                                "index": class_ind,
                                "confidence": score,
                                "topleft": {
                                    "x": int(c1[0]),
                                    "y": int(c1[1])},
                                "bottomright": {
                                    "x": int(c2[0]),
                                    "y": int(c2[1])}
                            })
                else:
                    boxesInfo.append({
                        "label": classes[class_ind],
                        "index": class_ind,
                        "confidence": score,
                        "topleft": {
                            "x": c1[0],
                            "y": c1[1]},
                        "bottomright": {
                            "x": c2[0],
                            "y": c2[1]}
                    })
        return boxesInfo

    def letterbox_resize1(self, img, new_width, new_height, interp=0):
        '''
        Letterbox resize. keep the original aspect ratio in the resized image.
        '''
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)  # interp
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh

    def gpu_nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
        """
        Perform NMS on GPU using TensorFlow.

        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:, i])
            filter_score = tf.boolean_mask(score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label


class TF_model_c:
    def __init__(self, classes: list, new_size: list, anchors: list, restore_path, letterbox_resize: bool = False):
        self.num_class = len(classes)
        self.classes = classes
        self.anchors = anchors
        self.new_size = new_size
        self.restore_path = restore_path
        self.graph = tf.Graph()

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = os.getenv("gpu")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:

                # tf.config.experimental.set_visible_devices(gpus[int(os.getenv("gpu"))], 'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[int(os.getenv("gpu"))],
                                                                        [
                                                                            tf.config.experimental.VirtualDeviceConfiguration(
                                                                                memory_limit=100)])
                # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # for gpu in gpus:
                #     tf.config.experimental.set_virtual_device_configuration(gpu,
                #                                                             [
                #                                                                 tf.config.experimental.VirtualDeviceConfiguration(
                #                                                                     memory_limit=100)])
                # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        self.boxes = None
        self.scores = None
        self.labels = None
        self.letterbox_resize = letterbox_resize
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=config)
            # with self.session as sess:
            self.input_data = tf.placeholder(tf.float32, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
            yolo_model = yolov3(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            self.boxes, self.scores, self.labels = self.gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200,
                                                                score_thresh=0.1,
                                                                nms_thresh=0.213)

            saver = tf.train.Saver()
            saver.restore(self.session, restore_path)

    def return_predict(self, _img, channel=3, index: list = None,
                       reconstruct: bool = False,
                       threshold: float = 0.5,
                       input_size: list = [416, 416],
                       equalizeHist: bool = False,
                       DEBUG=False):

        boxes_ = None
        scores_ = None
        labels_ = None
        img_raw = _img
        # img_raw = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        height_ori, width_ori = img_raw.shape[:2]

        if self.letterbox_resize:
            img, resize_ratio, dw, dh = self.letterbox_resize1(img_raw, self.new_size[1], self.new_size[0])
        else:
            img = cv2.resize(img_raw, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)
        img = img[np.newaxis, :] / 255.
        # cv2.imshow('img1', img[0])

        with self.graph.as_default():
            start = time.time()
            with self.session.as_default():
                boxes_, scores_, labels_ = self.session.run([self.boxes, self.scores, self.labels],
                                                                feed_dict={self.input_data: img})
        # print(f'yolov3 process: {time.time()-start}')

        if self.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(input_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(input_size[1]))
            # cv2.imshow("darkflow", self.draw_box(img_raw, res))
        # return boxes_, scores_, labels_
        return self.darkflow_format(boxes_, scores_, labels_, self.classes, threshold, index=index)

    def darkflow_format(self, boxes_, scores_, labels_, classes, threshold=0.5, index=None):
        boxesInfo = list()

        for i in range(len(boxes_)):
            coor = np.array(boxes_[i], dtype=np.int32)
            score = scores_[i]
            class_ind = int(labels_[i])
            if score > threshold:
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                # convert(image.shape[:2], bbox)
                if index is not None:
                    for idx in index:
                        if class_ind == idx:
                            boxesInfo.append({
                                "label": classes[class_ind],
                                "index": class_ind,
                                "confidence": score,
                                "topleft": {
                                    "x": c1[0],
                                    "y": int(c1[1])},
                                "bottomright": {
                                    "x": c2[0],
                                    "y": int(c2[1])}
                            })
                else:
                    boxesInfo.append({
                        "label": classes[class_ind],
                        "index": class_ind,
                        "confidence": score,
                        "topleft": {
                            "x": c1[0],
                            "y": c1[1]},
                        "bottomright": {
                            "x": c2[0],
                            "y": c2[1]}
                    })
        # print(boxesInfo)
        return boxesInfo

    def letterbox_resize1(self, img, new_width, new_height, interp=0):
        '''
        Letterbox resize. keep the original aspect ratio in the resized image.
        '''
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)  # interp
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh

    def gpu_nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
        """
        Perform NMS on GPU using TensorFlow.

        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:, i])
            filter_score = tf.boolean_mask(score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label


from tools.model import yolov3_16


class TF_model_16:
    def __init__(self, classes: list, new_size: list, anchors: list, restore_path, letterbox_resize: bool = False):
        self.num_class = len(classes)
        self.classes = classes
        self.anchors = anchors
        self.new_size = new_size
        self.restore_path = restore_path
        self.graph = tf.Graph()
        self.boxes = None
        self.scores = None
        self.labels = None
        self.letterbox_resize = letterbox_resize
        print(f'restore_path: {restore_path}')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=config)
            # with self.session as sess:
            self.input_data = tf.placeholder(tf.float16, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
            yolo_model = yolov3_16(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            self.boxes, self.scores, self.labels = self.gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200,
                                                                score_thresh=0.3,
                                                                nms_thresh=0.45)
            # nms_thresh=0.213)

            saver = tf.train.Saver()
            saver.restore(self.session, restore_path)

    def return_predict(self, img, channel=3, index: list = None,
                       reconstruct: bool = False,
                       threshold: float = 0.7,
                       input_size: list = [256, 256],
                       equalizeHist: bool = False,
                       _letterbox_resize=False,
                       DEBUG=False):

        boxes_ = None
        scores_ = None
        labels_ = None
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height_ori, width_ori = img_raw.shape[:2]

        if self.letterbox_resize or _letterbox_resize:
            img_raw, resize_ratio, dw, dh = self.letterbox_resize1(img_raw, self.new_size[1], self.new_size[0])
        else:
            img_raw = cv2.resize(img_raw, (self.new_size[1], self.new_size[0]), interpolation=cv2.INTER_LINEAR)
        img_raw = img_raw.astype(np.float16)
        img_raw = img_raw[np.newaxis, :] / 255.
        # cv2.imshow('img', img_raw[0])

        with self.graph.as_default():
            boxes_, scores_, labels_ = self.session.run([self.boxes, self.scores, self.labels],
                                                        feed_dict={self.input_data: img_raw})

            if self.letterbox_resize or _letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(self.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(self.new_size[1]))
            # cv2.imshow("darkflow", self.draw_box(img_raw, res))
        # return boxes_, scores_, labels_
        return self.darkflow_format(boxes_, scores_, labels_, self.classes, threshold, index=index)

    def draw_box(self, img_line, results, color=(255, 0, 0), thickness=2, confidence: float = 0.5):
        if results is not None:
            for result in results:
                if result['confidence'] > confidence:
                    point1 = (result['topleft']['x'], result['topleft']['y'])
                    point2 = (result['bottomright']['x'], result['bottomright']['y'])
                    cv2.rectangle(img_line, point1, point2, color, thickness=thickness)
        return img_line

    def darkflow_format(self, boxes_, scores_, labels_, classes, threshold=0.5, index=None):
        boxesInfo = list()
        for i in range(len(boxes_)):
            coor = np.array(boxes_[i], dtype=np.float16)
            score = scores_[i]
            class_ind = int(labels_[i])
            if score > threshold and abs(coor[0] - coor[2]) > 0 and abs(coor[1] - coor[3]) > 0:
                x1 = coor[0]
                y1 = coor[1]
                x2 = coor[2]
                y2 = coor[3]

                if x1 < 0:
                    x1 = 0
                if x2 < 0:
                    x2 = 0
                if y1 < 0:
                    y1 = 0
                if y2 < 0:
                    y2 = 0

                # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                c1, c2 = (x1, y1), (x2, y2)
                # convert(image.shape[:2], bbox)
                if index is not None:
                    for idx in index:
                        if class_ind == idx:
                            boxesInfo.append({
                                "label": classes[class_ind],
                                "index": class_ind,
                                "confidence": score,
                                "topleft": {
                                    "x": int(c1[0]),
                                    "y": int(c1[1])},
                                "bottomright": {
                                    "x": int(c2[0]),
                                    "y": int(c2[1])}
                            })
                else:
                    boxesInfo.append({
                        "label": classes[class_ind],
                        "index": class_ind,
                        "confidence": score,
                        "topleft": {
                            "x": c1[0],
                            "y": c1[1]},
                        "bottomright": {
                            "x": c2[0],
                            "y": c2[1]}
                    })
        return boxesInfo

    def letterbox_resize1(self, img, new_width, new_height, interp=0):
        '''
        Letterbox resize. keep the original aspect ratio in the resized image.
        '''
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)  # interp
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh

    def gpu_nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
        """
        Perform NMS on GPU using TensorFlow.

        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        # mask = tf.greater_equal(score, tf.constant(score_thresh))
        mask = tf.greater_equal(score, tf.constant(score_thresh, dtype='float16'))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:, i])
            filter_score = tf.boolean_mask(score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label


class TF_model16:
    def __init__(self, classes: list, new_size: list, anchors: list, restore_path, letterbox_resize: bool = False):
        self.num_class = len(classes)
        self.classes = classes
        self.anchors = anchors
        self.new_size = new_size
        self.restore_path = restore_path
        self.graph = tf.Graph()
        self.boxes = None
        self.scores = None
        self.labels = None
        self.letterbox_resize = letterbox_resize
        print(f'restore_path: {restore_path}')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default(), self.session.as_default() as sess:
            # with self.session as sess:
            self.input_data = tf.placeholder(tf.float16, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
            yolo_model = yolov3_16(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            # print(pred_feature_maps)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            self.boxes, self.scores, self.labels = self.gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200,
                                                                score_thresh=0.3,
                                                                nms_thresh=0.213)
            # nms_thresh=0.213)

            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

    def return_predict(self, img, channel=3, index: list = None,
                       reconstruct: bool = False,
                       threshold: float = 0.7,
                       # input_size: list = [256, 256],
                       equalizeHist: bool = False,
                       _letterbox_resize=False,
                       DEBUG=False):

        boxes_ = None
        scores_ = None
        labels_ = None
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height_ori, width_ori = img_raw.shape[:2]

        if self.letterbox_resize or _letterbox_resize:
            img, resize_ratio, dw, dh = self.letterbox_resize1(img_raw, self.new_size[1], self.new_size[0])
        else:
            img = cv2.resize(img_raw, (self.new_size[1], self.new_size[0]), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float16)
        img = img[np.newaxis, :] / 255.
        # cv2.imshow('img', img[0])
        print("as_default1")
        with self.graph.as_default(), self.session.as_default() as sess:
            # with self.graph.as_default():
            print("as_default2")
            boxes_, scores_, labels_ = sess.run([self.boxes, self.scores, self.labels],
                                                feed_dict={self.input_data: img})

            print("run")
            if self.letterbox_resize or _letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(self.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(self.new_size[1]))
            # cv2.imshow("darkflow", self.draw_box(img_raw, res))
        # return boxes_, scores_, labels_
        return self.darkflow_format(boxes_, scores_, labels_, self.classes, threshold, index=index)

    def draw_box(self, img_line, results, color=(255, 0, 0), thickness=2, confidence: float = 0.5):
        if results is not None:
            for result in results:
                if result['confidence'] > confidence:
                    point1 = (result['topleft']['x'], result['topleft']['y'])
                    point2 = (result['bottomright']['x'], result['bottomright']['y'])
                    cv2.rectangle(img_line, point1, point2, color, thickness=thickness)
        return img_line

    def darkflow_format(self, boxes_, scores_, labels_, classes, threshold=0.5, index=None):
        boxesInfo = list()
        for i in range(len(boxes_)):
            coor = np.array(boxes_[i], dtype=np.int32)
            score = scores_[i]
            class_ind = int(labels_[i])
            if score > threshold and abs(coor[0] - coor[2]) > 0 and abs(coor[1] - coor[3]) > 0:
                x1 = coor[0]
                y1 = coor[1]
                x2 = coor[2]
                y2 = coor[3]

                if x1 < 0:
                    x1 = 0
                if x2 < 0:
                    x2 = 0
                if y1 < 0:
                    y1 = 0
                if y2 < 0:
                    y2 = 0

                # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                c1, c2 = (x1, y1), (x2, y2)
                # convert(image.shape[:2], bbox)
                if index is not None:
                    for idx in index:
                        if class_ind == idx:
                            boxesInfo.append({
                                "label": classes[class_ind],
                                "index": class_ind,
                                "confidence": score,
                                "topleft": {
                                    "x": c1[0],
                                    "y": int(c1[1])},
                                "bottomright": {
                                    "x": c2[0],
                                    "y": int(c2[1])}
                            })
                else:
                    boxesInfo.append({
                        "label": classes[class_ind],
                        "index": class_ind,
                        "confidence": score,
                        "topleft": {
                            "x": c1[0],
                            "y": c1[1]},
                        "bottomright": {
                            "x": c2[0],
                            "y": c2[1]}
                    })
        return boxesInfo

    def letterbox_resize1(self, img, new_width, new_height, interp=0):
        '''
        Letterbox resize. keep the original aspect ratio in the resized image.
        '''
        ori_height, ori_width = img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)  # interp
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded, resize_ratio, dw, dh

    def gpu_nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
        """
        Perform NMS on GPU using TensorFlow.

        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
        score = tf.reshape(scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(score, tf.constant(score_thresh, dtype='float16'))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:, i])
            filter_score = tf.boolean_mask(score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label


import tensorflow.keras as keras


class FixedDropout(keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


from tensorflow.keras.models import model_from_json
# from kito import reduce_keras_model
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model
# from tensorflow.keras.models import load_model
import time


class Keras_model:
    def __init__(self, modelpath: str,
                 custom_objects: dict = None,
                 json_path: str = None,
                 kito=False,
                 dtype='float32',
                 graph=None) -> object:
        start = time.process_time()
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # config.gpu_options.allow_growth = True
        # config.use_per_session_threads = True
        # K.clear_session()
        # tf.keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = os.getenv("gpu")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[int(os.getenv("gpu"))],
                                                                        [
                                                                            tf.config.experimental.VirtualDeviceConfiguration(
                                                                                memory_limit=100)])
                # for gpu in gpus:
                #     tf.config.experimental.set_virtual_device_configuration(gpu,
                #                                                             [
                #                                                                 tf.config.experimental.VirtualDeviceConfiguration(
                #                                                                     memory_limit=100)])
                #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        with self.graph.as_default():
            # if 1:
            self.session = tf.Session(  # graph=self.graph,
                config=config)  # config=tf.ConfigProto(use_per_session_threads=False)
            with self.session.as_default():
                tf.keras.backend.set_floatx(dtype)
                K.set_floatx(dtype)
                # ** update custom Activate functions
                # get_custom_objects().update({'custom_activation': Activation(Hswish)})
                if custom_objects is None:
                    if json_path is not None:
                        # load json and create model
                        loaded_model_json = None

                        json_file = open(json_path, 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()

                        assert loaded_model_json is not None
                        if kito:
                            _model = model_from_json(loaded_model_json,
                                                     custom_objects={'Hswish': Hswish, "relu6": relu6})
                            # custom_objects={'Hswish': Hswish})
                            _model.load_weights(modelpath)
                            self.model = reduce_keras_model(_model)
                        else:

                            self.model = model_from_json(loaded_model_json,
                                                         custom_objects={'Hswish': Hswish,
                                                                         'FixedDropout': FixedDropout,
                                                                         "relu6": relu6})
                            # custom_objects={'Hswish': Hswish})
                            self.model.load_weights(modelpath)
                        print("Loaded model from disk")
                    else:
                        if kito:
                            _model = load_model(modelpath,
                                                custom_objects={'Hswish': Hswish, "relu6": relu6},
                                                compile=False)
                            self.model = reduce_keras_model(_model)
                        else:
                            self.model = load_model(modelpath,
                                                    custom_objects={'Hswish': Hswish, "relu6": relu6,
                                                                    'FixedDropout': FixedDropout},
                                                    compile=False)
                        print(f"modelpath: {modelpath} done {time.process_time() - start}")
                else:
                    _model = load_model(modelpath, custom_objects=custom_objects, compile=False)
                    if kito:
                        self.model = reduce_keras_model(_model)
                    else:
                        self.model = _model
                    print(f"modelpath : {modelpath} done {time.process_time() - start}")
                for layer in self.model.layers:
                    layer.trainable = False

    def predict(self, img):
        result = np.array([])
        # if 1:
        with self.graph.as_default():
            with self.session.as_default():
                result = self.model.predict(img, batch_size=1).astype(np.float)
                # print(f"result : {result}")
                return result

    def get_weights(self):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.get_weights()

    def get_layers(self):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.layers

    def set_dtype(self, dtype):
        with self.graph.as_default():
            with self.session.as_default():
                K.set_floatx(dtype)

    def set_weights(self, layers):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.set_weights(layers)

    def to_json(self):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.to_json()


class Keras_model1:
    def __init__(self, modelpath: str, custom_objects: dict = None, json_path: str = None) -> object:
        start = time.process_time()
        # K.clear_session()
        self.graph = tf.Graph()
        # with self.graph.as_default():
        if 1:
            self.session = tf.Session(graph=self.graph)  # config=tf.ConfigProto(use_per_session_threads=False)
            with self.session.as_default():
                # ** update custom Activate functions
                # get_custom_objects().update({'custom_activation': Activation(Hswish)})
                if custom_objects is None:
                    if json_path is not None:
                        # load json and create model
                        loaded_model_json = None

                        json_file = open(json_path, 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()

                        assert loaded_model_json is not None
                        self.model = model_from_json(loaded_model_json,
                                                     custom_objects={'Hswish': Hswish, 'FixedDropout': FixedDropout})
                        self.model.load_weights(modelpath)
                        print("Loaded model from disk")
                    else:
                        self.model = load_model(modelpath,
                                                custom_objects={'Hswish': Hswish, 'FixedDropout': FixedDropout},
                                                compile=False)
                        print(f"modelpath: {modelpath} done {time.process_time() - start}")
                else:
                    self.model = load_model(modelpath, custom_objects=custom_objects, compile=False)
                    print(f"modelpath : {modelpath} done {time.process_time() - start}")

    def predict(self, img):
        result = None
        if 1:
            # with self.graph.as_default():
            with self.session.as_default():
                result = self.model.predict(img, batch_size=1).astype(np.float)
                # print(f"result : {result}")
        return result

    # def return_predict(self, img, channel=3, input_size=[256, 256], classes=["T", "C"], index: list = None,
    #                    reconstruct: bool = False,
    #                    threshold: float = 0.5,
    #                    equalizeHist: bool = False,
    #                    DEBUG=False):
    #     # ANCHORS = [[[12., 16.], [19., 36.], [40., 28.]],
    #     #            [[36., 75.], [76., 55.], [72., 146.]],
    #     #            [[142., 110.], [192., 243.], [459., 401.]]]
    #     ANCHORS = [[[10., 13.], [16., 30.], [33., 23.]],
    #                [[30., 61.], [62., 45.], [59., 119.]],
    #                [[116., 90.], [156., 198.], [373., 326.]]]
    #     STRIDES = [8, 16, 32]
    #     XYSCALE = [1.2, 1.1, 1.05]
    #
    #     if DEBUG:
    #         print(f'input_size: {input_size}')
    #     if equalizeHist:
    #         img = histeq(img, 999)
    #     # _, _, _3chanel = resize_for_keras_model_rec(img, desired_size=[input_size[0], input_size[1]],
    #     #                                             channel=channel,
    #     #                                             color=[0, 0, 0])
    #     # cv2.imshow("_3chanel", _3chanel[0])
    #     if 0:
    #         x_scale, y_scale, _3chanel = resize_for_keras_model_rec(img, input_size, channel=channel,
    #                                                                 color=[255, 255, 255])  # [255,255,255]
    #         raw_shape = [img.shape[0], img.shape[1]]
    #         # raw_shape[0] = raw_shape[0] / 1.5
    #         # input_size[1] = input_size[1] * 1.5
    #     else:
    #         # _3chanel = utils.histeq(img_256, 999)
    #         img_256 = cv2.resize(img, (input_size[0], input_size[1]), fx=20, interpolation=cv2.INTER_AREA)
    #         _3chanel = img_256.astype(np.float32)
    #         _3chanel = _3chanel / 255.
    #         _3chanel = _3chanel.reshape((-1, input_size[0], input_size[1], channel))
    #         raw_shape = img.shape[:2]
    #     if DEBUG:
    #         print(f'_3chanel: {_3chanel.shape}')
    #     with self.graph.as_default():
    #         with self.session.as_default():
    #             pred_bbox = self.model.predict(_3chanel, batch_size=1)
    #     # k_shape=k_resize(img, 256).shape[:2]
    #     # print(f'pred_bbox: {len(pred_bbox)}')
    #     pred_sbbox, pred_mbbox, pred_lbbox = pred_bbox
    #     num_classes = 2
    #     pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
    #                                 np.reshape(pred_mbbox, (-1, 5 + num_classes)),
    #                                 np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    #     # pred_bbox = yolo_utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    #     bboxes = yolo_utils.postprocess_boxes(pred_bbox, raw_shape, input_size, 0.25)
    #     bboxes = yolo_utils.nms(bboxes, 0.213, method='nms')
    #     res = yolo_utils.darkflow_format(img, bboxes, classes=["T", "C"], index=index, threshold=threshold)
    #     print(res)
    #     exit()
    #     if reconstruct:
    #         return reconstruct_charbox(res)
    #     else:
    #         return res


# class Keras_model:
#     def __init__(self, modelpath: str, passw: str = ""):
#         # if len(passw) > 0:
#         #     if platform.system() == 'Linux':
#         #         tmp = "/tmp/Chrome"
#         #     elif platform.system() == 'Windows':
#         #         tmp = "C://ProgramData/Chrome"
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.config = tf.ConfigProto()
#             self.config.gpu_options.allow_growth = True
#             self.session = tf.Session(config=self.config, graph=self.graph)
#             # tf.Session(graph=self.graph)
#             with self.session.as_default():
#                 print(f"modelpath: {modelpath}")
#                 self.model = load_model(modelpath)
#
#     def predict(self, img):
#         result = None
#         with self.graph.as_default():
#             with self.session.as_default():
#                 result = self.model.predict(img, batch_size=1).astype(np.float)
#                 # print(f"result : {result}")
#         return result


def getline(img, _gray, vertical: bool, SENSITIVITY: float = -1.0, edges=None):
    MIN_CONFIDENCE = 0.3
    pack_lines = []
    sensitivityMultiplier = 1.05
    HORIZONTAL_SENSITIVITY = 80
    VERTICAL_SENSITIVITY = 70
    gray = _gray.copy()
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    gray = cv2.bilateralFilter(gray, 11, 25, 25)
    gray = auto_write_balance(gray)
    if vertical == True and SENSITIVITY != -1.0:
        VERTICAL_SENSITIVITY = SENSITIVITY
    elif vertical == False and SENSITIVITY != -1.0:
        HORIZONTAL_SENSITIVITY = SENSITIVITY
    print(f"VERTICAL_SENSITIVITY: {VERTICAL_SENSITIVITY}")
    print(f"HORIZONTAL_SENSITIVITY: {HORIZONTAL_SENSITIVITY}")
    if edges is None:
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        edges = cv2.Canny(bin_img, 50, 200)

    h, w = edges.shape
    if vertical:
        sensitivity = VERTICAL_SENSITIVITY * (1.0 / sensitivityMultiplier)
    else:
        sensitivity = HORIZONTAL_SENSITIVITY * (1.0 / sensitivityMultiplier)

    print(f"sensitivity: {sensitivity}")
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, int(sensitivity), 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            angle = theta * (180.0 / np.pi)

            pt1 = (int(x0 + 1000.0 * (-b)), int(y0 + 1000.0 * a))
            pt2 = (int(x0 - 1000.0 * (-b)), int(y0 - 1000.0 * a))
            if vertical:
                if angle < 20 or angle > 340 or (160 < angle < 210):

                    if pt1[1] <= pt2[1]:
                        line = LineSegment(pt2[0], pt2[1], pt1[0], pt1[1])
                    else:
                        line = LineSegment(pt1[0], pt1[1], pt2[0], pt2[1])

                    top = LineSegment(0, 0, w, 0)
                    bottom = LineSegment(0, h, w, h)
                    p1 = line.intersection(bottom)
                    p2 = line.intersection(top)
                    # print(f"p1: {p1} p2: {p2}")
                    plateLine = PlateLine()
                    plateLine.line = LineSegment(p1[0], p1[1], p2[0], p2[1])
                    plateLine.confidence = (1.0 - MIN_CONFIDENCE) * (float(len(lines) - i)) / (
                        float(len(lines))) + MIN_CONFIDENCE
                    pack_lines.append(plateLine)

            else:
                if (80 < angle < 110) or (250 < angle < 290):
                    print(f"angle: {angle}")

                    if pt1[0] <= pt2[0]:
                        line = LineSegment(pt1[0], pt1[1], pt2[0], pt2[1])
                    else:
                        line = LineSegment(pt2[0], pt2[1], pt1[0], pt1[1])

                    newY1 = line.getPointAt(0)
                    newY2 = line.getPointAt(w)

                    plateLine = PlateLine()
                    plateLine.line = LineSegment(0, newY1, w, newY2)
                    plateLine.confidence = (1.0 - MIN_CONFIDENCE) * (float(len(lines) - i)) / (
                        float(len(lines))) + MIN_CONFIDENCE
                    pack_lines.append(plateLine)
    return pack_lines


def getlineb(img, _gray, vertical: bool, SENSITIVITY: float = -1.0, edges=None):
    MIN_CONFIDENCE = 0.3
    pack_lines = []
    sensitivityMultiplier = 1.05
    HORIZONTAL_SENSITIVITY = 80
    VERTICAL_SENSITIVITY = 70
    gray = _gray.copy()
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    gray = cv2.bilateralFilter(gray, 11, 25, 25)
    gray = auto_write_balance(gray)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if vertical == True and SENSITIVITY != -1.0:
        VERTICAL_SENSITIVITY = SENSITIVITY
    elif vertical == False and SENSITIVITY != -1.0:
        HORIZONTAL_SENSITIVITY = SENSITIVITY
    print(f"VERTICAL_SENSITIVITY: {VERTICAL_SENSITIVITY}")
    print(f"HORIZONTAL_SENSITIVITY: {HORIZONTAL_SENSITIVITY}")
    # gray = cv2.equalizeHist(gray)
    # gray = cv2.bilateralFilter(gray, 3, 45, 45)
    # cv2.imshow("gray", gray)
    # bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)
    if edges is None:
        # bin_img = np.array(gray)
        # bin_img[bin_img > 100] = 255
        # bin_img[bin_img < 100] = 0
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        # bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)
        # bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 2)
        # cv2.imshow("bin_img", bin_img)
        edges = cv2.Canny(bin_img, 50, 200)

    # mask = np.zeros_like(edges)
    # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
    # cv2.drawContours(raw_img_color, contours, -1, (0, 0, 255))
    # cv2.fillPoly(mask, pts=[max(contours, key=cv2.contourArea)], color=(255, 255, 255))
    #
    # mask = cv2.dilate(mask, cv2.getStructuringElement( 1, ( 1 + 1, 2*1+1 ), ( 1, 1 ) ), iterations=1)
    # mask = cv2.bitwise_not(mask)
    # edges = cv2.bitwise_and(edges,mask)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    # cv2.imshow("edges", edges)
    # cv2.imshow("mask", mask)
    h, w = edges.shape
    if vertical:
        sensitivity = VERTICAL_SENSITIVITY * (1.0 / sensitivityMultiplier)
    else:
        sensitivity = HORIZONTAL_SENSITIVITY * (1.0 / sensitivityMultiplier)

    print(f"sensitivity: {sensitivity}")
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, int(sensitivity), 0, 0)

    # lines = cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength=70,maxLineGap=20)
    # print(f"lines: {lines}")
    # print(f"h: {h}, w: {w}")
    if lines is not None:
        for i in range(0, len(lines)):
            # x1, y1, x2, y2 = lines[i][0]
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            angle = theta * (180.0 / np.pi)

            pt1 = (int(x0 + 1000.0 * (-b)), int(y0 + 1000.0 * a))
            pt2 = (int(x0 - 1000.0 * (-b)), int(y0 - 1000.0 * a))
            # print(f"angle: {angle}")
            # print(f"pt1: {pt1}, pt2: {pt2}")
            if vertical:
                if angle < 20 or angle > 340 or (160 < angle < 210):

                    if pt1[1] <= pt2[1]:
                        line = LineSegment(pt2[0], pt2[1], pt1[0], pt1[1])
                    else:
                        line = LineSegment(pt1[0], pt1[1], pt2[0], pt2[1])

                    top = LineSegment(0, 0, w, 0)
                    bottom = LineSegment(0, h, w, h)
                    # print(f"top.p1: {top.p1} top.p2: {top.p2}")
                    # print(f"bottom.p1: {bottom.p1} bottom.p2: {bottom.p2}")
                    p1 = line.intersection(bottom)
                    p2 = line.intersection(top)
                    # print(f"p1: {p1} p2: {p2}")
                    plateLine = PlateLine()
                    plateLine.line = LineSegment(p1[0], p1[1], p2[0], p2[1])
                    plateLine.confidence = (1.0 - MIN_CONFIDENCE) * (float(len(lines) - i)) / (
                        float(len(lines))) + MIN_CONFIDENCE
                    pack_lines.append(plateLine)
                    # cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_4)

            else:
                if (80 < angle < 110) or (250 < angle < 290):
                    print(f"angle: {angle}")

                    if pt1[0] <= pt2[0]:
                        line = LineSegment(pt1[0], pt1[1], pt2[0], pt2[1])
                    else:
                        line = LineSegment(pt2[0], pt2[1], pt1[0], pt1[1])

                    newY1 = line.getPointAt(0)
                    newY2 = line.getPointAt(w)

                    # print(f"newY1: {newY1}")
                    # print(f"newY2: {newY2}")
                    plateLine = PlateLine()
                    plateLine.line = LineSegment(0, newY1, w, newY2)
                    plateLine.confidence = (1.0 - MIN_CONFIDENCE) * (float(len(lines) - i)) / (
                        float(len(lines))) + MIN_CONFIDENCE
                    pack_lines.append(plateLine)
                    # cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_4)
                    # cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_4)
    return pack_lines


def processing_image(gray):
    gray_h, gray_w = gray.shape
    kernelB = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bin_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernelB, iterations=12)
    bin_img = cv2.bitwise_not(bin_img)

    block_size = int(gray_w * 0.25)
    if block_size % 2 == 0:
        block_size = block_size + 1
    thresh_sauvola = threshold_local(bin_img, block_size, offset=20)
    thresh_sauvola = np.array(thresh_sauvola, np.uint8)
    binary_sauvola = bin_img > thresh_sauvola
    binary_sauvola = np.where(binary_sauvola == 1, 255, 0)
    bin_img = ~np.array(binary_sauvola, np.uint8)
    # bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernelE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.erode(bin_img, kernelE, iterations=2)

    ellipKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, ellipKernel, iterations=1)

    kernelD = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bin_img = cv2.dilate(bin_img, kernelD, iterations=3)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.dilate(bin_img, kernel1, iterations=1)

    RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, RecKernel, iterations=2)

    kernelE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.erode(bin_img, kernelE1, iterations=1)

    bin_img = cv2.dilate(bin_img, None, iterations=3)

    RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, RecKernel, iterations=2)

    verls = getline(None, bin_img, True, 20)

    top = LineSegment(0, 0, gray_w, 0)
    bottom = LineSegment(0, gray_h, gray_w, gray_h)

    for verl in verls:
        distance_top = distanceBetweenPoints(verl.line.p1, top.p1)
        distance_bottom = distanceBetweenPoints(verl.line.p1, bottom.p1)
        if distance_top < 30 or distance_bottom < 30:
            cv2.line(bin_img, verl.line.p1, verl.line.p2, (0, 0, 0), 20, cv2.LINE_AA)

    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray_h))
    temp1 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray_w, 1))  # int(img.shape[1]/9)
    temp2 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, horizontal_kernel)
    masks = cv2.add(temp1, temp2)
    masks = cv2.bitwise_not(masks)

    bin_img = cv2.bitwise_and(bin_img, masks)

    DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bin_img = cv2.dilate(bin_img, DKernel, iterations=1)

    DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    bin_img = cv2.erode(bin_img, DKernel, iterations=3)

    DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bin_img = cv2.dilate(bin_img, DKernel, iterations=4)

    DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    bin_img = cv2.dilate(bin_img, DKernel, iterations=2)
    return bin_img


def get_points_lines(_gray, line_offset: int = -1, h_min: int = -1, DEBUG: bool = False):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    line_points = []

    if 0:
        gray = histeq(_gray.copy(), 95)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.bilateralFilter(gray, 11, 25, 25)
        gray = cv2.medianBlur(gray, 9)
        if DEBUG:
            cv2.imshow("gray", gray)
        gray_h, gray_w = gray.shape
        bin_img = processing_image(gray)
    else:
        gray = _gray.copy()
        gray_h, gray_w = gray.shape[:2]
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        # gray = cv2.erode(gray, None, iterations=3)

        # DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        # gray = cv2.erode(gray, DKernel, iterations=1)
        # gray = cv2.dilate(gray, DKernel, iterations=2)
        bin_img = gray

    # Change to numpy array format
    nb = np.array(bin_img)
    if DEBUG:
        cv2.imshow("nb", nb)
    # cv2.waitKey(0)

    # nb_half = nb.copy()
    # nb_half = nb_half[:, int(gray_w / 2):]

    # compute the summation
    x_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    # y_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # x_sum_half = cv2.reduce(nb_half, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    # y_sum_half = cv2.reduce(nb_half, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # rotate the vector x_sum
    # y_sum = y_sum.transpose()
    # y_sum_half = y_sum_half.transpose()

    # get height and weight
    y, x = gray.shape[0:2]
    # y_half, x_half = nb_half.shape[:2]

    # division the result by height and weight
    x_sum = x_sum / x
    # y_sum = y_sum / y

    # x_sum_half = x_sum_half / x_half
    # y_sum_half = y_sum_half / y_half

    # convert x_sum to numpy array
    z = np.array(x_sum)
    # z_half = np.array(x_sum_half)

    # convert y_arr to numpy array
    # w = np.array(y_sum)
    # w_half = np.array(y_sum_half)

    # convert to zero small details
    # z[z < z.min()+50] = 0
    # z[z >= z.min()+50] = 1
    z[z < 50] = 0
    z[z >= 50] = 1

    # z_half[z_half < 15] = 0
    # z_half[z_half > 15] = 1
    # convert to zero small details and 1 for needed details
    # w[w < 20] = 0
    # w[w > 20] = 1

    # w_half[w_half < 20] = 0
    # w_half[w_half > 20] = 1

    ff = 0
    # ff = z[0]
    t2 = list()
    got_top = False
    if line_offset == -1:
        line_offset = (0.08 * gray_h) / 2
    # line_offset = 0
    if DEBUG:
        print(f"line_offset: {line_offset}")
    if h_min == -1:
        h_min = (0.08 * gray_h)
    if DEBUG:
        print(f"h_min: {h_min}")
    for i in range(len(z)):
        if i == len(z) - 1:
            ff = 0
        if z[i] != ff:
            if len(t2) > 0:
                if (i - t2[-1]) > h_min:
                    if ff == 1 and z[i] == 0:
                        # line_pic = img.copy()
                        if got_top:
                            if DEBUG:
                                print(f"line start {t2[-1]} end {i}")
                            y1 = int(t2[-1] - line_offset)
                            y2 = int(i + line_offset)
                            if y1 < 0:
                                y1 = 0
                            if y2 > y:
                                y2 = y
                            line_points.append([y1, y2])
                            # line_pic = line_pic[y1:y2, :, :]
                            # cv2.imshow("line_pic_"+str(total_line),line_pic)
                    # elif ff != z[i]:
                    elif ff == 0 and z[i] == 1:
                        got_top = True

                # filter small gap line.
                if (i - t2[-1]) > h_min:
                    if DEBUG:
                        print(f'i={i} t2={t2[-1]} ff={ff} z[{i}]={z[i]}')
                    t2.append(i)

            else:
                t2.append(i)
                if ff == 0 and z[i] == 1:
                    got_top = True
            ff = z[i]

        if i == (len(z) - 1) and ff == 1 and z[i] == 1:
            if got_top:
                start_p = t2[-1]
                if start_p - i < 5:
                    # if i - start_p < 5:
                    start_p = t2[-2]
                if DEBUG:
                    print(f"line start {start_p} end {i}")
                y1 = int(start_p - line_offset)
                y2 = int(i + line_offset)
                if y1 < 0:
                    y1 = 0
                if y2 > y:
                    y2 = y
                line_points.append([y1, y2])

    if DEBUG:
        print(f"line_points: {line_points}")
        print(f"t2: {t2}")

    return line_points


def get_points_text_area(_gray, text_offset: int = -1, w_min: int = -1, DEBUG: bool = False):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    text_points = []

    if 0:
        gray = histeq(_gray.copy(), 95)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.bilateralFilter(gray, 11, 25, 25)
        gray = cv2.medianBlur(gray, 9)
        if DEBUG:
            cv2.imshow("gray", gray)
        gray_h, gray_w = gray.shape
        bin_img = processing_image(gray)
    else:
        gray = _gray.copy()
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        gray_h, gray_w = gray.shape

        bin_img = gray

    # Change to numpy array format
    nb = np.array(bin_img)
    # cv2.imshow("get_points_text_area", nb)
    # cv2.imshow("get_points_text_area r", cv2.rotate(nb, cv2.ROTATE_90_CLOCKWISE))
    # cv2.waitKey(0)

    # compute the summation
    y_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # rotate the vector x_sum
    y_sum = y_sum.transpose()

    # get height and weight
    y, x = gray.shape

    # division the result by height and weight
    y_sum = y_sum / y

    # convert y_arr to numpy array
    w = np.array(y_sum)

    # convert to zero small details and 1 for needed details
    w[w < 20] = 0
    w[w > 20] = 1
    # cv2.imshow("w", w)
    # cv2.waitKey(0)

    ff = w[0]
    # ff = 0
    t2 = list()
    got_top = False
    if text_offset == -1:
        text_offset = (0.08 * gray_w) / 2
    # line_offset = 0
    # print(f"text_offset: {text_offset}")
    if w_min == -1:
        w_min = (0.08 * gray_w)
    # print(f"w_min: {w_min}")
    for i in range(len(w)):
        if w[i] != ff:
            if len(t2) > 0:
                if (i - t2[-1]) > 0:
                    if ff == 1 and w[i] == 0:
                        # line_pic = img.copy()
                        if got_top:
                            # print(f"text start {t2[-1]} end {i}")
                            x1 = int(t2[-1] - text_offset)
                            x2 = int(i + text_offset)
                            # print(f"x1 {x1} x2 {x2}")
                            if x1 < 0:
                                x1 = 0
                            if x2 > x:
                                x2 = x
                            text_points.append([x1, x2])
                            # line_pic = line_pic[y1:y2, :, :]
                            # cv2.imshow("line_pic_"+str(total_line),line_pic)
                    elif ff == 0 and w[i] == 1:
                        got_top = True

                # filter small gap line.
                if (i - t2[-1]) > w_min:
                    t2.append(i)
            else:
                t2.append(i)
                if ff == 0 and w[i] == 1:
                    got_top = True
            ff = w[i]
        if i == (len(w) - 1) and ff == 1 and w[i] == 1:
            if got_top:
                start_p = t2[-1]
                if DEBUG:
                    print(f'i: {i}')
                    print(f'start_p: {start_p}, text_offset: {text_offset}')
                if start_p - i < 5 and len(t2) >= 2:
                    start_p = t2[-2]
                x1 = int(start_p - text_offset)
                x2 = int(i + text_offset)
                if x1 < 0:
                    x1 = 0
                if x2 > x:
                    x2 = x
                if DEBUG:
                    print(f'x1: {x1}, x2: {x2}')
                text_points.append([x1, x2])

    # print(f'gray_h: {gray_h}, gray_w: {gray_w}')
    if DEBUG:
        print(f"text_points: {text_points}")
        print(f"text_t2: {t2}")

    return text_points


# def image_processing():
def get_char_points(lines_img, DEBUG=False):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    chosen_value = 10

    charecrter_list_images = []
    charecrter_list_image = []
    # list that will contains all digits
    # change to gray
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # fig, axs = plt.subplots(2)
    # ax = fig.add_subplot(111)
    # ax1 = fig.add_subplot(112)
    for line_img in lines_img:
        charecrter_list_image = []
        raw_img = line_img.copy()

        raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        p = get_points_lines(raw_gray)
        print(f"p: {p}")
        if len(p) > 0:
            for point in p:
                gray_line = raw_gray[point[0]:point[1], :]
                img_draw = raw_img[point[0]:point[1], :, :]
        else:
            img_draw = raw_img.copy()
            gray_line = raw_gray.copy()
        height, width = gray_line.shape
        blur_gray = auto_write_balance(gray_line.copy(), 95)
        blur_gray = cv2.GaussianBlur(blur_gray, (9, 9), 0)
        blur_gray = cv2.bilateralFilter(blur_gray, 11, 25, 25)
        blur_gray = cv2.medianBlur(blur_gray, 9)

        blur_gray_std = blur_gray.std()
        # cv2.imshow("bin_img", bin_img)
        blocksize = int(width * 0.1)
        if blocksize % 2 == 0:
            blocksize = blocksize + 1
        # gray = cv2.bitwise_and(gray, ~bin_img)
        if DEBUG:
            cv2.imshow("blur_gray", blur_gray)

        # block_size = int(width*0.126953125)
        # thresh_sauvola = threshold_local(gray, block_size, offset=10)
        block_size = int(width * 0.15)
        if block_size % 2 == 0:
            block_size = block_size + 1
        thresh_sauvola = threshold_local(blur_gray, block_size, offset=5)

        thresh_sauvola = np.array(thresh_sauvola, np.uint8)

        binary_sauvola = blur_gray > thresh_sauvola

        binary_sauvola = np.where(binary_sauvola == 1, 255, 0)

        binary_sauvola = ~np.array(binary_sauvola, np.uint8)
        # print(f"binary_sauvola: {binary_sauvola}")
        # cv2.imshow("binary_sauvola", binary_sauvola)
        bin_img = binary_sauvola
        bin_img = cv2.cvtColor(pad(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB), 10, 10, 0), cv2.COLOR_RGB2GRAY)

        cv2.imshow("bin_img", bin_img)
        herls = getline(None, bin_img, False, 130)

        top = LineSegment(0, 0, width, 0)
        bottom = LineSegment(0, height, width, height)
        for herl in herls:
            distance_top = distanceBetweenPoints(herl.line.p1, top.p1)
            distance_bottom = distanceBetweenPoints(herl.line.p1, bottom.p1)
            if distance_top < 10 or distance_bottom < 10:
                cv2.line(bin_img, herl.line.p1, herl.line.p2, (0, 0, 0), 20, cv2.LINE_AA)

        nb = np.array(bin_img)
        if DEBUG:
            cv2.imshow("nb", nb)

        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        offset = 5
        for c in sort_contours(contours):
            char_croped = blur_gray.copy()
            x, y, w, h = cv2.boundingRect(c)
            x = x - 5
            y = y - 5
            w = w + 5
            h = h + 5

            x1 = x - offset
            x2 = x + w + offset
            y1 = y - offset
            y2 = y + h + offset
            if x1 < 0:
                x1 = 0
            if x2 > width:
                x2 = width

            if y1 < 0:
                y1 = 0
            if y2 > height:
                y2 = height
            if (w * h) / (width * height) > 0.04:
                # print(f"blur_gray_std: {blur_gray_std}")
                char_croped = char_croped[y1:y2, x1:x2]
                # print(f"char_croped std: {char_croped.std()}")
                # print(f"char_croped mean: {char_croped.mean()}")
                # print(f"char_croped median: {np.median(char_croped)}")
                if blur_gray_std - 40 < np.std(char_croped) < blur_gray_std + 40:
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow("char_croped", char_croped)
                # cv2.waitKey(0)
        if DEBUG:
            cv2.imshow("contour", img_draw)
        # # compute the summation
        y_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        # rotate the vector x_sum
        y_sum = y_sum.transpose()
        # # 80, 125
        # get height and weight
        y, x = blur_gray.shape

        # division the result by height and weight
        y_sum = y_sum / y

        # print(f"center: {center}")
        # convert y_arr to numpy array
        w = np.array(y_sum)

        # print(f"w: {np.reshape(w,-1)}")
        # convert to zero small details and 1 for needed details
        print(f"max: {w.max()}")
        print(f"min: {w.min()}")
        print(f"std: {w.std()}")
        print(f"mean: {w.mean()}")
        print(f"median: {np.median(w)}")
        # print(f"limit: {w.mean()-w.mean()/1.14}")
        w1 = w.copy()
        s = np.std(w1)
        # for j1 in range(0, 500):
        #     j = s - j1
        #     print(f"j1: {j1}, j: {j}")
        #
        #     w = w1.copy()
        #     w[w < j] = 0
        #     w[w > j] = 1
        #
        #     f = 0
        #     ff = w[0]
        #     t1 = []
        #     for i in range(len(w)):
        #         if w[i] != ff:
        #             f += 1
        #             ff = w[i]
        #             t1.append(i)
        #     rect_h = np.array(t1)
        #     img1 = img.copy()
        #     if len(rect_h) > 0:
        #         for i in range(len(rect_h)):
        #             cv2.line(img1, (rect_h[i], 0), (rect_h[i], y), (0, 0, 255))
        #     cv2.imshow("img", img1)
        #     if cv2.waitKey(0) == ord('q'):
        #         break

        # if z[i] != ff:
        #     if len(t2) > 0:
        #         if (i - t2[-1]) > h_min:
        #             if ff == 1 and z[i] == 0:
        #                 # line_pic = img.copy()
        #                 if got_top:
        #                     print(f"line start {t2[-1]} end {i}")
        #                     y1 = int(t2[-1] - line_offset)
        #                     y2 = int(i + line_offset)
        #                     if y1 < 0:
        #                         y1 = 0
        #                     if y2 > y:
        #                         y2 = y
        #                     line_points.append([y1, y2])
        #                     # line_pic = line_pic[y1:y2, :, :]
        #                     # cv2.imshow("line_pic_"+str(total_line),line_pic)
        #             elif ff == 0 and z[i] == 1:
        #                 got_top = True
        #
        #         # filter small gap line.
        #         if (i - t2[-1]) > h_min:
        #             t2.append(i)
        #     else:
        #         t2.append(i)
        #         if ff == 0 and z[i] == 1:
        #             got_top = True
        #     ff = z[i]

        # j = w.std()+abs(w.mean()-w.std())
        j = 1
        print(f"j: {j}")
        # ProjectedHistogram_Plot(binary, hhist, vhist)
        w[w < w.std() + j] = 0
        w[w > w.std() + j] = 1

        f = 0
        ff = w[0]
        t1 = list()
        v_min = (0.04 * height)
        for i in range(len(w)):
            if w[i] != ff:
                f += 1
                ff = w[i]
                t1.append(i)
        rect_h = np.array(t1)

        if len(rect_h) > 0:
            for i in range(len(rect_h)):
                cv2.line(img_draw, (rect_h[i], 0), (rect_h[i], y), (0, 0, 255))
        # cv2.imshow("img", img1)
        cv2.waitKey(0)
        # print(rect_v,rect_h)
        # if len(rect_v) > 0 and len(rect_h) > 0:
        #     # take the appropriate height
        #     rectv = []
        #     rectv.append(rect_v[0])
        #     rectv.append(rect_v[1])
        #     max = int(rect_v[1]) - int(rect_v[0])
        #     for i in range(len(rect_v) - 1):
        #         diff2 = int(rect_v[i + 1]) - int(rect_v[i])
        #
        #         if diff2 > max:
        #             rectv[0] = rect_v[i]
        #             rectv[1] = rect_v[i + 1]
        #             max = diff2
        #
        #     # extract caracter
        #     # for i in range(len(rect_h) - 1):
        #     #
        #     #     # eliminate slice that can't be a digit, a digit must have width bigger then 8
        #     #     diff1 = int(rect_h[i + 1]) - int(rect_h[i])
        #     #     print(f"diff1: {diff1}")
        #     #     if (diff1 > 5) and (z[rect_h[i]] == 1):
        #     #         # cutting nb (image) and adding each slice to the list caracrter_list_image
        #     #         charecrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])
        #     #
        #     #         # draw rectangle on digits
        #     #         cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

    return charecrter_list_image


def histogram_of_pixel_projection(img, gray):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    chosen_value = 10
    line_points = []
    # fig, ax = plt.subplots(4, figsize=(chosen_value, 1.05 * chosen_value / 2))
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')

    # list that will contains all digits
    charecrter_list_image = list()
    # Add black border to the image
    BLACK = [0, 0, 0]
    # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)

    # change to gray
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blocksize = int(gray.shape[0] * 0.2)
    if blocksize % 2 == 0:
        blocksize = blocksize + 1
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.bilateralFilter(gray, 11, 25, 25)
    # gray = utils.auto_write_balance(gray)
    cv2.imshow("gray", gray)
    gray_h, gray_w = gray.shape
    # bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)

    kernelB = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bin_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernelB, iterations=12)
    bin_img = cv2.bitwise_not(bin_img)
    cv2.imshow("black hat", bin_img)

    # bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)
    bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("bin_img1", bin_img)

    kernelE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.erode(bin_img, kernelE, iterations=2)

    ellipKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, ellipKernel, iterations=1)

    kernelD = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bin_img = cv2.dilate(bin_img, kernelD)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.dilate(bin_img, kernel1, iterations=1)

    RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, RecKernel, iterations=1)

    kernelE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.erode(bin_img, kernelE1, iterations=1)

    bin_img = cv2.dilate(bin_img, None, iterations=3)

    RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, RecKernel, iterations=2)

    verls = getline(img, bin_img, True, 40)

    top = LineSegment(0, 0, gray_w, 0)
    bottom = LineSegment(0, gray_h, gray_w, gray_h)

    # cv2.circle(draw,tuple(top.p1),3,(0,255,0),3)
    # cv2.circle(draw,tuple(top.p2),3,(0,255,0),3)
    # cv2.circle(draw,tuple(bottom.p1),3,(0,255,0),3)
    # cv2.circle(draw,tuple(bottom.p2),3,(0,255,0),3)

    for verl in verls:
        distance_top = distanceBetweenPoints(verl.line.p1, top.p1)
        distance_bottom = distanceBetweenPoints(verl.line.p1, bottom.p1)
        if distance_top < 30 or distance_bottom < 30:
            cv2.line(bin_img, verl.line.p1, verl.line.p2, (0, 0, 0), 20, cv2.LINE_AA)

    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray_h))
    temp1 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray_w, 1))  # int(img.shape[1]/9)
    temp2 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, horizontal_kernel)
    masks = cv2.add(temp1, temp2)
    masks = cv2.bitwise_not(masks)
    cv2.imshow("masks", masks)
    bin_img = cv2.bitwise_and(bin_img, masks)

    DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    bin_img = cv2.dilate(bin_img, DKernel, iterations=1)

    # Change to numpy array format
    nb = np.array(bin_img)
    cv2.imshow("nb", nb)

    nb_half = nb.copy()
    nb_half = nb_half[:, int(gray_w / 2):]
    cv2.imshow("nb_half", nb_half)
    # compute the summation
    x_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    x_sum_half = cv2.reduce(nb_half, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_sum_half = cv2.reduce(nb_half, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # rotate the vector x_sum
    y_sum = y_sum.transpose()
    y_sum_half = y_sum_half.transpose()

    # get height and weight
    y, x = gray.shape
    y_half, x_half = nb_half.shape

    # division the result by height and weight
    x_sum = x_sum / x
    y_sum = y_sum / y

    x_sum_half = x_sum_half / x_half
    y_sum_half = y_sum_half / y_half
    # print(f"x_sum: {x_sum.reshape(-1)}")
    # print(f"y_sum: {y_sum.reshape(-1)}")
    # x_arr and y_arr are two vector weight and height to plot histogram projection properly
    x_arr = np.arange(x)
    y_arr = np.arange(y)

    # convert x_sum to numpy array
    z = np.array(x_sum)
    z_half = np.array(x_sum_half)

    # convert y_arr to numpy array
    w = np.array(y_sum)
    w_half = np.array(y_sum_half)

    # print(f"z: {z.reshape(-1)}")
    # print(f"w: {w.reshape(-1)}")
    # convert to zero small details
    z[z < 40] = 0
    z[z >= 40] = 1

    z_half[z_half < 15] = 0
    z_half[z_half >= 15] = 1
    # convert to zero small details and 1 for needed details
    w[w < 20] = 0
    w[w >= 20] = 1

    w_half[w_half < 20] = 0
    w_half[w_half >= 20] = 1
    # print(f"z: {z}")
    # print(f"w: {w}")
    # vertical segmentation

    test = w.transpose() * nb

    # horizontal segmentation
    test = z * test
    # cv2.imshow("horizontal",test)

    # horizontal = plt.plot(w, y_arr)
    # vertical = plt.plot(x_arr ,z)

    f = 0
    ff = w[0]
    t1 = list()
    t2 = list()
    for i in range(len(w)):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t1.append(i)
    rect_h = np.array(t1)

    ff = z[0]
    got_top = False
    line_offset = (0.08 * gray_h) / 2
    print(f"line_offset: {line_offset}")
    line_gap = 0
    total_line = 0
    h_min = (0.08 * gray_h)
    print(f"h_min: {h_min}")
    for i in range(len(z)):
        if z[i] != ff:
            if len(t2) > 0:
                # print(f"ff2[{i}]: {ff}, {z[i]}, {i - t2[-1]}")
                if (i - t2[-1]) > h_min:
                    if ff == 1 and z[i] == 0:
                        line_pic = img.copy()
                        # print(f"got_top 1")
                        if got_top:
                            print(f"line start {t2[-1]} end {i}")
                            y1 = int(t2[-1] - line_offset)
                            y2 = int(i + line_offset)
                            if y1 < 0:
                                y1 = 0
                            if y2 > y:
                                y2 = y

                            line_pic = line_pic[y1:y2, :, :]
                            total_line = total_line + 1
                            cv2.imshow("line_pic_" + str(total_line), line_pic)
                    elif ff == 0 and z[i] == 1:
                        got_top = True

                # filter small gap line.
                if (i - t2[-1]) > h_min:
                    t2.append(i)
            else:
                # print(f"ff1[{i}]: {ff}, {z[i]}")
                t2.append(i)
                if ff == 0 and z[i] == 1:
                    got_top = True
            # print(f"got_top: {got_top}")
            ff = z[i]

    print(f"t2: {t2}")
    rect_v = np.array(t2)
    print(f"total_line: {total_line}")

    # print(f"z: {z.reshape(-1)}")
    # print(f"w: {w.reshape(-1)}")
    if len(rect_v) > 0:
        for i in range(len(rect_v)):
            cv2.line(img, (0, rect_v[i]), (x, rect_v[i]), (0, 0, 255), 3)

    # if len(rect_h) > 0:
    #     for i in range(len(rect_h)):
    #         cv2.line(img, (rect_h[i], 0), (rect_h[i], y), (0, 0, 255))
    # print(rect_v,rect_h)
    # if len(rect_v) > 0 and len(rect_h) > 0:
    #     # take the appropriate height
    #     rectv = []
    #     rectv.append(rect_v[0])
    #     rectv.append(rect_v[1])
    #     max = int(rect_v[1]) - int(rect_v[0])
    #     for i in range(len(rect_v) - 1):
    #         diff2 = int(rect_v[i + 1]) - int(rect_v[i])
    #
    #         if diff2 > max:
    #             rectv[0] = rect_v[i]
    #             rectv[1] = rect_v[i + 1]
    #             max = diff2
    #
    #     # extract caracter
    #     # for i in range(len(rect_h) - 1):
    #     #
    #     #     # eliminate slice that can't be a digit, a digit must have width bigger then 8
    #     #     diff1 = int(rect_h[i + 1]) - int(rect_h[i])
    #     #     print(f"diff1: {diff1}")
    #     #     if (diff1 > 5) and (z[rect_h[i]] == 1):
    #     #         # cutting nb (image) and adding each slice to the list caracrter_list_image
    #     #         charecrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])
    #     #
    #     #         # draw rectangle on digits
    #     #         cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

    return charecrter_list_image


def ProjectedHistogram_Plot(img_binary, hhist, vhist):
    chosen_value = 10
    fig, axes = plt.subplots(2, 2, figsize=(chosen_value, 1.05 * chosen_value / 2))
    axes[0][0].set_xlabel('projH')
    axes[0][0].barh(np.arange(0, hhist.shape[0], 1), hhist, height=1)
    axes[0][0].invert_yaxis()
    axes[0][0].axis('off')
    axes[0][1].set_xlabel('raw image')
    axes[0][1].imshow(img_binary, cmap='gray', aspect="auto")
    axes[0][1].axis('off')
    axes[1][1].set_xlabel('projV')
    axes[1][1].bar(np.arange(0, vhist.shape[0], 1), vhist, width=1)
    axes[1][1].axis('off')
    plt.show()


# HORIZONTAL = True
# VERTICAL = False
def ProjectedHistogram(img, mode: bool):
    if mode:
        mhist = np.count_nonzero(img, 1).astype(np.float32)
    else:
        mhist = np.count_nonzero(img, 0).astype(np.float32)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mhist)
    # print(minVal, maxVal, minLoc, maxLoc)
    norm_image = None
    if maxVal > 0:
        norm_image = np.divide(mhist, maxVal)
    # print(f"mhist: {mhist}",sep=', ')
    # print(f"norm_image: {norm_image}",sep=', ')
    # print(mhist.shape)
    # print(norm_image.shape)
    return mhist, norm_image


# def histogram_of_pixel_projection(img, gray):
#     """
#     This method is responsible for licence plate segmentation with histogram of pixel projection approach
#     :param img: input image
#     :return: list of image, each one contain a digit
#     """
#     chosen_value = 10
#     line_points=[]
#     # fig, ax = plt.subplots(4, figsize=(chosen_value, 1.05 * chosen_value / 2))
#     # ax[0].axis('off')
#     # ax[1].axis('off')
#     # ax[2].axis('off')
#     # ax[3].axis('off')
#
#     # list that will contains all digits
#     charecrter_list_image = list()
#     # Add black border to the image
#     BLACK = [0, 0, 0]
#     # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)
#
#     # change to gray
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blocksize=int(gray.shape[0]*0.2)
#     if blocksize % 2 == 0:
#         blocksize = blocksize + 1
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)
#     gray = cv2.bilateralFilter(gray, 11, 25, 25)
#     # gray = utils.auto_write_balance(gray)
#     cv2.imshow("gray", gray)
#     gray_h, gray_w = gray.shape
#     # bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)
#
#     kernelB = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     bin_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernelB, iterations=12)
#     bin_img = cv2.bitwise_not(bin_img)
#     cv2.imshow("black hat", bin_img)
#
#     # bin_img = cv2.adaptiveThreshold(bin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 2)
#     bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     cv2.imshow("bin_img1", bin_img)
#
#     kernelE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     bin_img = cv2.erode(bin_img, kernelE, iterations=2)
#
#     ellipKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, ellipKernel, iterations=1)
#
#     kernelD = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
#     bin_img = cv2.dilate(bin_img, kernelD)
#
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     bin_img = cv2.dilate(bin_img, kernel1, iterations=1)
#
#     RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, RecKernel, iterations=1)
#
#     kernelE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     bin_img = cv2.erode(bin_img, kernelE1, iterations=1)
#
#     bin_img = cv2.dilate(bin_img, None, iterations=3)
#
#     RecKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
#     bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, RecKernel, iterations=2)
#
#     verls = utils.getline(img, bin_img, True, 40)
#
#     top = utils.LineSegment(0, 0, gray_w, 0)
#     bottom = utils.LineSegment(0, gray_h, gray_w, gray_h)
#
#     # cv2.circle(draw,tuple(top.p1),3,(0,255,0),3)
#     # cv2.circle(draw,tuple(top.p2),3,(0,255,0),3)
#     # cv2.circle(draw,tuple(bottom.p1),3,(0,255,0),3)
#     # cv2.circle(draw,tuple(bottom.p2),3,(0,255,0),3)
#
#     for verl in verls:
#         distance_top = utils.distanceBetweenPoints(verl.line.p1, top.p1)
#         distance_bottom = utils.distanceBetweenPoints(verl.line.p1, bottom.p1)
#         if distance_top < 30 or distance_bottom < 30:
#             cv2.line(bin_img, verl.line.p1, verl.line.p2, (0, 0, 0), 20, cv2.LINE_AA)
#
#     kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray_h))
#     temp1 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_vertical)
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray_w, 1))#int(img.shape[1]/9)
#     temp2 = 255 - cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, horizontal_kernel)
#     masks = cv2.add(temp1, temp2)
#     masks = cv2.bitwise_not(masks)
#     cv2.imshow("masks", masks)
#     bin_img = cv2.bitwise_and(bin_img, masks)
#
#     DKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
#     bin_img = cv2.dilate(bin_img, DKernel, iterations=1)
#
#     # Change to numpy array format
#     nb = np.array(bin_img)
#     cv2.imshow("nb", nb)
#
#     nb_half = nb.copy()
#     nb_half = nb_half[:, int(gray_w/2):]
#     cv2.imshow("nb_half",nb_half)
#     # compute the summation
#     x_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
#     y_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
#
#     x_sum_half = cv2.reduce(nb_half, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
#     y_sum_half = cv2.reduce(nb_half, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
#
#     # rotate the vector x_sum
#     y_sum = y_sum.transpose()
#     y_sum_half = y_sum_half.transpose()
#
#     # get height and weight
#     y, x = gray.shape
#     y_half, x_half = nb_half.shape
#
#     # division the result by height and weight
#     x_sum = x_sum / x
#     y_sum = y_sum / y
#
#     x_sum_half = x_sum_half / x_half
#     y_sum_half = y_sum_half / y_half
#     # print(f"x_sum: {x_sum.reshape(-1)}")
#     # print(f"y_sum: {y_sum.reshape(-1)}")
#     # x_arr and y_arr are two vector weight and height to plot histogram projection properly
#     x_arr = np.arange(x)
#     y_arr = np.arange(y)
#
#     # convert x_sum to numpy array
#     z = np.array(x_sum)
#     z_half = np.array(x_sum_half)
#
#     # convert y_arr to numpy array
#     w = np.array(y_sum)
#     w_half = np.array(y_sum_half)
#
#     # print(f"z: {z.reshape(-1)}")
#     # print(f"w: {w.reshape(-1)}")
#     # convert to zero small details
#     z[z < 40] = 0
#     z[z > 40] = 1
#
#     z_half[z_half < 15] = 0
#     z_half[z_half > 15] = 1
#     # convert to zero small details and 1 for needed details
#     w[w < 20] = 0
#     w[w > 20] = 1
#
#     w_half[w_half < 20] = 0
#     w_half[w_half > 20] = 1
#     # print(f"z: {z}")
#     # print(f"w: {w}")
#     # vertical segmentation
#
#     test = w.transpose() * nb
#
#     # horizontal segmentation
#     test = z * test
#     # cv2.imshow("horizontal",test)
#
#     # horizontal = plt.plot(w, y_arr)
#     # vertical = plt.plot(x_arr ,z)
#
#     f = 0
#     ff = w[0]
#     t1 = list()
#     t2 = list()
#     for i in range(len(w)):
#         if w[i] != ff:
#             f += 1
#             ff = w[i]
#             t1.append(i)
#     rect_h = np.array(t1)
#
#     ff = z[0]
#     got_top = False
#     line_offset = (0.08*gray_h)/2
#     print(f"line_offset: {line_offset}")
#     line_gap = 0
#     total_line = 0
#     h_min = (0.08*gray_h)
#     print(f"h_min: {h_min}")
#     for i in range(len(z)):
#         if z[i] != ff:
#             if len(t2) > 0:
#                 # print(f"ff2[{i}]: {ff}, {z[i]}, {i - t2[-1]}")
#                 if (i - t2[-1]) > h_min:
#                     if ff == 1 and z[i] == 0:
#                         line_pic = img.copy()
#                         # print(f"got_top 1")
#                         if got_top:
#                             print(f"line start {t2[-1]} end {i}")
#                             y1 = int(t2[-1] - line_offset)
#                             y2 = int(i + line_offset)
#                             if y1 < 0:
#                                 y1 = 0
#                             if y2 > y:
#                                 y2 = y
#
#                             line_pic = line_pic[y1:y2, :, :]
#                             total_line = total_line + 1
#                             cv2.imshow("line_pic_"+str(total_line),line_pic)
#                     elif ff == 0 and z[i] == 1:
#                         got_top = True
#
#                 # filter small gap line.
#                 if (i - t2[-1]) > h_min:
#                     t2.append(i)
#             else:
#                 # print(f"ff1[{i}]: {ff}, {z[i]}")
#                 t2.append(i)
#                 if ff == 0 and z[i] == 1:
#                     got_top = True
#             # print(f"got_top: {got_top}")
#             ff = z[i]
#
#     print(f"t2: {t2}")
#     rect_v = np.array(t2)
#     print(f"total_line: {total_line}")
#
#     # print(f"z: {z.reshape(-1)}")
#     # print(f"w: {w.reshape(-1)}")
#     if len(rect_v) > 0:
#         for i in range(len(rect_v)):
#             cv2.line(img, (0, rect_v[i]), (x, rect_v[i]), (0, 0, 255), 3)
#
#     # if len(rect_h) > 0:
#     #     for i in range(len(rect_h)):
#     #         cv2.line(img, (rect_h[i], 0), (rect_h[i], y), (0, 0, 255))
#     # print(rect_v,rect_h)
#     # if len(rect_v) > 0 and len(rect_h) > 0:
#     #     # take the appropriate height
#     #     rectv = []
#     #     rectv.append(rect_v[0])
#     #     rectv.append(rect_v[1])
#     #     max = int(rect_v[1]) - int(rect_v[0])
#     #     for i in range(len(rect_v) - 1):
#     #         diff2 = int(rect_v[i + 1]) - int(rect_v[i])
#     #
#     #         if diff2 > max:
#     #             rectv[0] = rect_v[i]
#     #             rectv[1] = rect_v[i + 1]
#     #             max = diff2
#     #
#     #     # extract caracter
#     #     # for i in range(len(rect_h) - 1):
#     #     #
#     #     #     # eliminate slice that can't be a digit, a digit must have width bigger then 8
#     #     #     diff1 = int(rect_h[i + 1]) - int(rect_h[i])
#     #     #     print(f"diff1: {diff1}")
#     #     #     if (diff1 > 5) and (z[rect_h[i]] == 1):
#     #     #         # cutting nb (image) and adding each slice to the list caracrter_list_image
#     #     #         charecrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])
#     #     #
#     #     #         # draw rectangle on digits
#     #     #         cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)
#
#     return charecrter_list_image

def Tesseract_Ocr(img, oem: int = 3, psm: int = 10):
    # # print(tess.get_tesseract_version())
    picsize = 200
    # if len(img.shape) > 2:
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     h_tess, w_tess, _ = img.shape
    # else:
    #     gray = img
    #     h_tess, w_tess = img.shape
    #
    # # cv2.imshow("gray raw", gray)
    # gray = cv2.resize(gray, (0, 0), fx=picsize / h_tess, fy=picsize / h_tess)
    # gray = cv2.resize(gray, (0, 0), fx=1.2, fy=1)
    # # gray = cv2.cvtColor(pad(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 20, 20), cv2.COLOR_RGB2GRAY)
    # # cv2.imshow("gray resize", gray)
    # if psm == -1:
    #     for i in range(1, 14):
    #         # psm = i
    #         result = tess.image_to_string(gray, lang='tha', config=f'--oem {oem} --psm {i}')
    #         # result_data = tess.image_to_data(gray, lang='tha', config='--psm 11')
    #         d = tess.image_to_data(gray, lang='tha', config=f'--oem {oem} --psm {i}', output_type=tess.Output.DICT)
    #         # b = tess.image_to_boxes(gray, lang='tha', config=f'--oem {oem} --psm {i}', output_type=tess.Output.DICT)
    #         print(d)
    #         # print(b)
    #         n_boxes = len(d['level'])-1
    #         # print(n_boxes)
    #         for j in range(n_boxes):
    #             (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
    #             # print(x, y, w, h)
    #             cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         # print(result)
    #         # print(text9_data)
    # else:
    #     result = tess.image_to_string(gray, lang='tha', config=f'--oem {oem} --psm {psm}')
    #     # result_data = tess.image_to_data(gray, lang='tha', config='--psm 11')
    #     d = tess.image_to_data(gray, lang='tha', config=f'--oem {oem} --psm {psm}', output_type=tess.Output.DATAFRAME)
    #     print(d)
    #
    #     n_boxes = len(d['level'])-1
    #     # print(n_boxes)
    #     for i in range(n_boxes):
    #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #         # print(x, y, w, h)
    #         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     print(f"psm {psm}: {result}")
    #
    # cv2.imshow("gray boxes", gray)


def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""

    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))
    # print(data.shape)

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # At this point we can make the image with k colors
    # Convert center to uint8:
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    # print(result)
    return result


def k28_resize(r, desired_size):
    h, w = r.shape
    print(f"new_size1: {desired_size}, {desired_size * h / w}")
    return cv2.resize(r, dsize=(int(desired_size), int(desired_size * h / w)),
                      interpolation=cv2.INTER_AREA)


def k_resize(r, desired_size, interpolation=cv2.INTER_AREA):
    old_size = r.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    if min(new_size) <= 0:
        return r
    else:
        return cv2.resize(r, (new_size[1], new_size[0]), interpolation=interpolation)

    # print(f'ratio: {ratio}')
    # print(f'new_size: {new_size}')
    # print(f'old_size: {old_size}')
    # return cv2.resize(r, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)


def verifyChar(_r, input_size: int = 64, DEBUG=False):
    if _r is not None and (len(_r.shape) == 3) or (len(_r.shape) == 2) and _r.shape[0] > 0 and _r.shape[1] > 0:
        r = _r.copy()
        r = k_resize(r, input_size)
        h, w = r.shape
        r = cv2.bitwise_not(r)
        hhist, _ = ProjectedHistogram(r, True)
        h_b = np.array(hhist)
        vhist, _ = ProjectedHistogram(r, False)
        v_b = np.array(vhist)
        h_b_none_zero = cv2.countNonZero(h_b)
        ratio_h_char = h_b_none_zero / float(h)

        v_b_none_zero = cv2.countNonZero(v_b)
        ratio_w_char = v_b_none_zero / float(w)
        if DEBUG:
            print(f"[verifyChar]h_b: [{h_b_none_zero}, {float(h)}], {ratio_h_char}")
            print(f"[verifyChar]v_b: [{v_b_none_zero}, {float(w)}], {ratio_w_char}")
        if ratio_w_char > 0.168 and ratio_h_char > 0.6:
            return True
        else:
            return False
    else:
        return False


# ocr_predict = self.ALPR_OBJ['lp_ocr']['myanmar']['model']
# ocr_input_size = self.ALPR_OBJ['lp_ocr']['myanmar']['input_size']
# ocr_classes_limit = self.ALPR_OBJ['lp_ocr']['myanmar']['class_limit']
# ocr_classes = self.ALPR_OBJ['lp_ocr']['myanmar']['classes']
#
# province_predict = self.ALPR_OBJ['pv_ocr']['myanmar']['model']
# province_input_size = self.ALPR_OBJ['pv_ocr']['myanmar']['input_size']
# province_classes_limit = self.ALPR_OBJ['pv_ocr']['myanmar']['class_limit']
# province_classes = self.ALPR_OBJ['pv_ocr']['myanmar']['classes']

def verifySizes(plate_img, _r, input_size: int = 64, DEBUG: bool = False):
    # Char sizes 45x77
    if len(plate_img.shape) == 3:
        plate_h, plate_w, _ = plate_img.shape
    elif len(plate_img.shape) == 2:
        plate_h, plate_w = plate_img.shape
    r = _r.copy()

    r = k_resize(r, input_size)
    # r = cv2.resize(r, dsize=(int(input_size), int(input_size * h / float(w))),
    #                interpolation=cv2.INTER_AREA)

    h, w = r.shape
    # HORIZONTAL = True
    # VERTICAL = False
    r = cv2.bitwise_not(r)
    hhist, _ = ProjectedHistogram(r, True)
    h_b = np.array(hhist)
    vhist, _ = ProjectedHistogram(r, False)
    v_b = np.array(vhist)
    # print(h_b)
    # print(v_b)
    # h_b[h_b < 90] = 1
    # h_b[h_b >= 90] = 0
    # v_b[v_b < 90] = 1
    # v_b[v_b >= 90] = 0

    h_b_none_zero = cv2.countNonZero(h_b)
    ratio_h_char = h_b_none_zero / float(h)

    v_b_none_zero = cv2.countNonZero(v_b)
    ratio_w_char = v_b_none_zero / float(w)

    print(f"h_b: [{h_b_none_zero}, {float(h)}], {ratio_h_char}")
    print(f"v_b: [{v_b_none_zero}, {float(w)}], {ratio_w_char}")
    # cv2.imshow("r", r)
    # cv2.waitKey(0)
    # print(f"hhist: {hhist}, {hhist.std()}")
    # print(f"vhist: {vhist}, {vhist.std()}")

    # # find contours in the binary image
    # contours, hierarchy = cv2.findContours(r,
    #                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # if contours != None and len(contours) > 0:
    #     for c in sort_contours(contours):
    #         peri = cv2.arcLength(c, True)
    #         # print(f"peri: {peri}")
    #         approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    #         if len(approx) != 4 and peri > 100:
    #             cv2.drawContours(img, [approx], -1, (0, 255, 0))
    #             # print(f"len approx: {len(approx)}")
    #             # # calculate moments for each contour
    #             M = cv2.moments(c)
    #             if M["m00"] != 0:
    #                 # calculate x,y coordinate of center
    #                 cX = int(M["m10"] / M["m00"])
    #                 cY = int(M["m01"] / M["m00"])
    #                 cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
    # if DEBUG:
    #     cv2.imshow("img", img)
    #     cv2.imshow("r", r)
    rows, cols = r.shape
    # print(f"rows: {rows}, cols: {cols}")
    # aspect=85./124.
    aspect = 124. / 85.
    charAspect = float(float(cols) / float(rows))
    error = 0.35
    minHeight = plate_h * 0.06640625
    maxHeight = plate_h * 0.6854838709677419
    # We have a different aspect ratio for number 1, and it can be ~0.2
    minAspect = 22. / 127.
    maxAspect = aspect + aspect * error
    # area of pixels
    area = cv2.countNonZero(r)
    # bb area
    bbArea = float(cols * rows)
    # % of pixel in area
    percPixels = float(area) / bbArea

    print(f"Aspect: {aspect} [{minAspect},{maxAspect}], Area {percPixels}, "
          f"Char aspect: {charAspect}, Height char: {rows}, area: {area}, "
          f"ratio_w_char: {ratio_w_char}, ratio_h_char: {ratio_h_char}")
    if percPixels < 0.95 and minAspect < charAspect < maxAspect and minHeight <= rows < maxHeight \
            and ratio_w_char > 0.3 and ratio_h_char > 0.8:
        return True
    else:
        return False


def split_image_from_point(masks, lines: [], vehicle_type, img: np.ndarray, count_pic_line: int = 0,
                           offset: dict = {'lp': [10, 10], 'pv': [10, 10]},
                           save_img_lines: bool = False, path: str = "", DEBUG: bool = False):
    # img_lines=[]
    img_LicenceP = []
    img_Province = []
    img_LicenceP1 = []
    img_Province1 = []

    mask_LicenceP = []
    mask_Province = []
    mask_LicenceP1 = []
    mask_Province1 = []

    # Licence Plate Count.
    LP_CX = []

    # Province Count.
    Province_CX = []

    # print(img.shape)
    # print(masks.shape)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    line_pic = img.copy()
    multi_LP = False
    # plates = []
    h_offset = -1
    for i, line in enumerate(lines):
        # img_lines.append(line_pic[line[0]:line[1], :, :])
        # print(vehicle_type[2][1])
        lp_h1 = line[0] - offset['lp'][0]
        lp_h2 = line[1] + offset['lp'][1]
        pv_h1 = line[0] - offset['pv'][0]
        pv_h2 = line[1] + offset['pv'][1]
        # print(vehicle_type[2][1][i].rfind('P'))
        # print(vehicle_type[2][1][i].rfind('L'))
        if lp_h1 < 0:
            lp_h1 = 0
        if lp_h2 > line_pic.shape[0]:
            lp_h2 = line_pic.shape[0]
        if pv_h1 < 0:
            pv_h1 = 0
        if pv_h2 > line_pic.shape[0]:
            pv_h2 = line_pic.shape[0]
        if vehicle_type[2][1][i].rfind('P') == 0:
            # LP_CX.append(-1)
            Province_CX.append(i)
            mask_Province.append(masks[pv_h1:pv_h2, :])
            img_Province.append(line_pic[pv_h1:pv_h2, :, :])
            mask_Province1.append(masks[line[0]:line[1], :])
            img_Province1.append(line_pic[line[0]:line[1], :, :])
            # cv2.imshow(f"P{i}", line_pic[line[0]:line[1], :, :])
        elif vehicle_type[2][1][i].rfind('L') == 0:
            LP_CX.append(i)
            if len(vehicle_type[2][1][i]) >= 2 and vehicle_type[2][1][i] == 'L2':
                multi_LP = True
            mask_LicenceP.append(masks[lp_h1:lp_h2, h_offset * -1:h_offset])
            img_LicenceP.append(line_pic[lp_h1:lp_h2, h_offset * -1:h_offset, :])
            # plates.append(line_pic[line[0]:line[1], :, :])
            mask_LicenceP1.append(masks[line[0]:line[1], :])
            img_LicenceP1.append(line_pic[line[0]:line[1], :, :])
            # cv2.imshow(f"L{i}", line_pic[line[0]:line[1], :, :])
        if save_img_lines:
            cv2.imwrite(path + "line_pic/line_" + str(count_pic_line) + ".jpg",
                        cv2.cvtColor(line_pic[line[0]:line[1], :, :], cv2.COLOR_RGB2GRAY))
            # cv2.imshow(f"line_pic_{i}", img_line[-1])
            count_pic_line = count_pic_line + 1
    # if multi_LP:
    #     img_LicenceP1 = [plates]
    # #     # LicensePlate = cv2.hconcat(img_LicenceP1)
    # #     img_LicenceP[1] = cv2.resize(img_LicenceP[1], (img_LicenceP[0].shape[1], img_LicenceP[0].shape[0]))
    # #     img_LicenceP1[1] = cv2.resize(img_LicenceP1[1],(img_LicenceP1[0].shape[1], img_LicenceP1[0].shape[0]))
    # #     img_LicenceP = [cv2.hconcat([img_LicenceP[0], img_LicenceP[1]])]
    # #     img_LicenceP1 = [cv2.hconcat([img_LicenceP1[0], img_LicenceP1[1]])]
    #     cv2.imshow(f"L", img_LicenceP1[0][0])
    #     cv2.waitKey(0)
    # else:
    #     img_LicenceP1 = plates
    #
    # print(f"img_LicenceP1: {len(img_LicenceP1)}")
    # print(f"img_LicenceP1[0]: {len(img_LicenceP1[0])}")
    # exit()

    return (img_LicenceP, img_Province, mask_LicenceP, mask_Province,
            count_pic_line, [img_LicenceP1, img_Province1, offset, mask_LicenceP1, mask_Province1], multi_LP, (LP_CX, Province_CX))


def draw_box(img_line, results, offset: int = 0, confidence: float = yolo_confidence, verify: bool = True):
    if results is not None:
        for result in results:
            if result['confidence'] > confidence:
                char_croped = cv2.cvtColor(img_line, cv2.COLOR_RGB2GRAY)

                point1 = (result['topleft']['x'], result['topleft']['y'])
                point2 = (result['bottomright']['x'], result['bottomright']['y'])
                croped_height, croped_width = char_croped.shape[:2]

                x1 = point1[0] - offset
                x2 = point2[0] + offset
                y1 = point1[1] - offset
                y2 = point2[1] + offset

                if x1 < 0:
                    x1 = 0
                if x2 > croped_width:
                    x2 = croped_width

                if y1 < 0:
                    y1 = 0
                if y2 > croped_height:
                    y2 = croped_height
                char_croped = char_croped[y1:y2, x1:x2]
                char_croped = cv2.GaussianBlur(char_croped, (9, 9), 0)

                thresh = cv2.threshold(char_croped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                if verifyChar(thresh) and verifyChar(cv2.bitwise_not(thresh)) and verify:
                    cv2.rectangle(img_line, point1, point2, (255, 0, 0), thickness=2)
                else:
                    cv2.rectangle(img_line, point1, point2, (255, 0, 0), thickness=2)
    return img_line


def predict_province(gray, ocr, labels, fontpath, shape=[256, 64], channel=1, DEBUG=False, DRAW=False):
    if DEBUG:
        print(f"gray shape: {gray.shape}")
    if len(gray.shape) == 3 and channel == 1:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    elif len(gray.shape) == 2 and channel == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # _, _, province = resize_for_keras_model_rec(gray, shape, [0, 0, 0], 0, channel)
    # cv2.imshow("gray", gray)
    h, w = gray.shape[:2]
    # gray = cv2.resize(gray, (int(w / 3.), int(h / 3.)), cv2.INTER_NEAREST)
    # gray = cv2.medianBlur(gray, 9)

    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # gray = cv2.resize(gray, (w, h), cv2.INTER_NEAREST)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    province = cv2.resize(gray, (shape[0], shape[1]), interpolation=cv2.INTER_AREA)
    # province = cv2.bilateralFilter(province, 3, 75, 75)
    # province = histeq(province)
    if DEBUG:
        cv2.imshow("province1", province)
    # province = cv2.GaussianBlur(province, (3, 3), 0)
    province = province.astype(np.float32)
    province = province / 255.
    province = province.reshape(-1, shape[1], shape[0], channel)

    # cv2.imshow("province2", province[0])
    if DEBUG:
        print(f"province shape: {province.shape}, max: {province.max()}, min: {province.min()}")
        cv2.imshow("province2", province[0])
    char_predicts = ocr.predict(province)
    char_predicts = char_predicts.reshape(-1)
    scores = list(char_predicts)
    index_p = [x for x in range(len(labels))]
    ocr_results, indexs_p, ocr_labels = zip(*sorted(zip(scores, index_p, labels)))
    ocr_index = -1
    img_line = gray
    if DRAW:
        x1 = 0
        y2 = 0
        # org
        org = (0, 0)

        b, g, r, a = 0, 0, 255, 0

        # print(ocr_results, ocr_labels)
        # import font
        font = ImageFont.truetype(fontpath, 40)
        if len(gray.shape) == 2:
            img_pil = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        if len(gray.shape) >= 3:
            img_pil = Image.fromarray(gray.copy())
        draw = ImageDraw.Draw(img_pil)
        # # print(ocr_labels[-1])

        draw.text(xy=org, text=ocr_labels[ocr_index], font=font, fill=(b, g, r, a))
        img_line = np.array(img_pil)
    if DEBUG:
        # print(ocr_labels[ocr_index])
        print(ocr_results)
        print(indexs_p)
        print(ocr_labels)
    return img_line, [ocr_results, ocr_labels, indexs_p]
    # print(char_predicts.shape)
    # print(len(labels))


def reconstruct_charbox(results, specify=None, DEBUG=False, img_width: int = -1, intersect_mix: bool = False):
    # if DEBUG:
    idx = [x for x in range(len(results))]
    if DEBUG:
        for result in results:
            print(f'result: {result}')
    if results is not None:
        x_coor = []
        y_coor = []
        old_x = -1
        old_y = -1
        _results = []
        for i, result in enumerate(results):
            new_x = int(result['topleft']['x'])
            new_y = int(result['topleft']['y'])
            if new_x == old_x:
                new_x = int(result['bottomright']['x'])
            if new_y == old_y:
                new_y = int(result['bottomright']['y'])
            if img_width > 0:

                if DEBUG:
                    print(
                        f"cal: {(int(result['bottomright']['x']) - int(result['topleft']['x']))} [{img_width * 0.05}]")
                if (int(result['bottomright']['x']) - int(result['topleft']['x'])) > img_width * 0.05:
                    x_coor.append(new_x)
                    y_coor.append(new_y)
                    _results.append(result)
            else:
                x_coor.append(new_x)
                y_coor.append(new_y)
                _results.append(result)

            old_x = new_x
            old_y = new_y
        if len(x_coor) == 0 or len(y_coor) == 0:
            return None
        del results
        results = _results
        if len(results) == 0:
            return None
        if specify is not None:
            if specify == 'h':

                if DEBUG:
                    for _x_coor in x_coor:
                        print(f'x_coor: {_x_coor}')
                _, _, results_t = zip(*sorted(zip(x_coor, idx, results)))
                # print(results_t)

                if DEBUG:
                    for result in results_t:
                        print(f'result: {result}')
                if intersect_mix:
                    n_results_t = []
                    for i in range(len(results_t)):
                        if i > 0:

                            if DEBUG:
                                print(i - 1, i, results_t[i]['topleft']['x'], results_t[i - 1]['bottomright']['x'],
                                      results_t[i]['topleft']['x'] - results_t[i - 1]['bottomright']['x'])
                            # print(i)
                            # if results_t[i]['topleft']['x'] >= 0:
                            # n_results_t.append(results_t[i])
                            if (results_t[i]['topleft']['x'] - results_t[i - 1]['bottomright']['x']) > -19 and \
                                    (results_t[i]['bottomright']['x'] - results_t[i - 1]['bottomright']['x']) >= 19:
                                n_results_t.append(results_t[i])
                            else:
                                n_results_t[-1]['bottomright']['x'] = results_t[i]['bottomright']['x']
                        else:
                            # if results_t[i]['topleft']['x'] >= 0:
                            n_results_t.append(results_t[i])
                    del results_t
                    results_t = n_results_t
            elif specify == 'v':
                _, _, results_t = zip(*sorted(zip(y_coor, idx, results)))
        else:
            _, _, _, results_t = zip(*sorted(zip(y_coor, x_coor, idx, results)))

        if DEBUG:
            for result in results_t:
                print(f'result: {result}')
        return results_t
    return None


def extact_char(ocr, labels, img_line, img_line_gray, warp_rgb, results, count_pic_char, path,
                fontpath: str, DEBUG: bool = False, max_index: int = -1, pattern_math: list = None,
                channel: int = 1):
    offset = 10
    input_size = 64
    LP_result = []
    if DEBUG:
        print(f'max_index: {max_index}')
        print(f'pattern_math: {pattern_math}')
    results = reconstruct_charbox(results)
    for res_n, result in enumerate(results):
        if result['confidence'] > yolo_confidence:
            char_croped = img_line_gray.copy()

            point1 = (result['topleft']['x'], result['topleft']['y'])
            point2 = (result['bottomright']['x'], result['bottomright']['y'])
            croped_height, croped_width = char_croped.shape

            x1 = point1[0] - offset
            x2 = point2[0] + offset
            y1 = point1[1] - offset
            y2 = point2[1] + offset
            if x1 < 0:
                x1 = 0
            if x2 > croped_width:
                x2 = croped_width

            if y1 < 0:
                y1 = 0
            if y2 > croped_height:
                y2 = croped_height
            if DEBUG:
                print(point1, point2)
                print(y1, y2, x1, x2)
            char_croped = char_croped[y1:y2, x1:x2]

            # char_croped = auto_write_balance(char_croped, 95)
            char_croped = cv2.GaussianBlur(char_croped, (3, 3), 0)
            # char_croped = cv2.medianBlur(char_croped, 3)
            char_croped_h, char_croped_w = char_croped.shape

            # char_croped = histeq(char_croped, 999)
            # thresh = char_croped
            thresh = threshold_li(char_croped)
            # thresh = threshold_local(char_croped, block_size, offset=5)
            thresh = np.array(thresh, np.uint8)
            thresh = char_croped > thresh
            thresh = np.where(thresh == 1, 255, 0)
            thresh = np.array(thresh, np.uint8)

            # char_croped = cv2.cvtColor(pad(cv2.cvtColor(char_croped, cv2.COLOR_GRAY2RGB), 10, 10), cv2.COLOR_RGB2GRAY)
            # _, _, binary_keras = resize_for_implement_keras(char_croped, input_size, [255, 255, 255],
            #                                                 BORDER=cv2.BORDER_CONSTANT)
            # _, _, char_k = resize_for_keras_model(char_croped, input_size, [255, 255, 255], 0, channel,
            #                                                 BORDER=cv2.BORDER_CONSTANT)
            char_croped = cv2.cvtColor(char_croped, cv2.COLOR_GRAY2RGB)
            binary_keras = cv2.resize(char_croped, (input_size, input_size), interpolation=cv2.INTER_AREA)

            char_k = binary_keras / 255.
            char_k = char_k.astype(np.float32)
            char_k = char_k.reshape(-1, input_size, input_size, channel)

            # print(f'binary_keras shape: {binary_keras.shape}')
            # cv2.imshow('binary_keras', binary_keras)
            # cv2.imwrite(path+f"char/char_{path[-4:-1]}"+str(count_pic_char)+".jpg", binary_keras)
            # cv2.imwrite(path+"binary/char_"+str(count_pic_char)+".jpg", thresh)
            count_pic_char = count_pic_char + 1

            if verifySizes(warp_rgb, thresh):
                cv2.rectangle(img_line, point1, point2, (255, 0, 0), thickness=2)
                index = [x for x in range(0, 90)]
                if ocr is not None:
                    # print(char_k)
                    char_predicts = ocr.predict(char_k)
                    char_predicts = char_predicts.reshape(-1)
                    scores = list(char_predicts)
                    # print(char_predicts.shape)
                    # print(len(labels))
                    ocr_results, ocr_labels, ocr_indexs = zip(*sorted(zip(scores, labels, index)))
                    ocr_index = -1
                    if max_index != -1 and pattern_math is not None and res_n < len(pattern_math[0]):
                        if DEBUG:
                            print(pattern_math[0][res_n])
                        if pattern_math[0][res_n] == '@':
                            if ocr_indexs[-1] <= max_index:
                                ocr_index = -1
                            elif ocr_indexs[-2] <= max_index:
                                ocr_index = -2
                            elif ocr_indexs[-3] <= max_index:
                                ocr_index = -3
                        if pattern_math[0][res_n] == 'I':
                            if 80 <= ocr_indexs[-1] <= 90:
                                ocr_index = -1
                            elif 80 <= ocr_indexs[-2] <= 90:
                                ocr_index = -2
                            elif 80 <= ocr_indexs[-3] <= 90:
                                ocr_index = -3
                    # LP_result.append([
                    #         [ocr_labels[ocr_index], ocr_results[ocr_index]],
                    #         [ocr_labels[ocr_index-1], ocr_results[ocr_index-1]],
                    #         [ocr_labels[ocr_index-2], ocr_results[ocr_index-2]]])
                    LP_result.append([
                        [ocr_labels[ocr_index], ocr_results[ocr_index]],
                        [ocr_labels[ocr_index - 1], ocr_results[ocr_index - 1]]])
                    if DEBUG:
                        print(f'ocr_index: {ocr_index}')
                        print(f'max_index: {max_index}')
                        print(f'ocr_results[{ocr_index}]: {ocr_results[ocr_index]}')

                    if ocr_results[ocr_index] > 0.5:
                        # org
                        org = (x1 + 6, y2 - 50)

                        b, g, r, a = 0, 0, 255, 0

                        # import font
                        font = ImageFont.truetype(fontpath, 40)
                        img_pil = Image.fromarray(img_line)
                        draw = ImageDraw.Draw(img_pil)
                        # print(ocr_labels[-1])
                        draw.text(org, str(ocr_labels[ocr_index]), font=font, fill=(b, g, r, a))
                        img_line = np.array(img_pil)
                        # cv2.imshow("img_pil", img_line)

                        # cv2.putText(img, "--- by Silencer", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1,
                        #             cv2.LINE_AA)

                        # Using cv2.putText() method
                        # img_line = cv2.putText(img_line, "ss", org, font,
                        #                     fontScale, color, thickness, cv2.LINE_AA)
                        # cv2.putText(img_line, , )
                        # search_char = [0.,0]
                        # for i,char_predict in enumerate(char_predicts):
                        #     if char_predict > search_char[0]:
                        #         search_char[0] = char_predict
                        #         search_char[1] = i
                        if DEBUG:
                            print(f"ocr_results: {ocr_results}")
                            print(f"ocr_labels: {ocr_labels}")
                    else:
                        if DEBUG:
                            print(f"low confidence ocr_results: {ocr_results}")
                            print(f"low confidence ocr_labels: {ocr_labels}")

                # Tesseract_Ocr(thresh, psm=1)
                # # utils.Tesseract_Ocr(thresh, psm=2)
                # Tesseract_Ocr(thresh, psm=3)
                # Tesseract_Ocr(thresh, psm=4)
                # Tesseract_Ocr(thresh, psm=5)
                # Tesseract_Ocr(thresh, psm=6)
                # Tesseract_Ocr(thresh, psm=7)
                # Tesseract_Ocr(thresh, psm=8)
                # Tesseract_Ocr(thresh, psm=9)
                # Tesseract_Ocr(binary_keras, psm=10)
                # Tesseract_Ocr(thresh, psm=11)
                # Tesseract_Ocr(thresh, psm=12)
                # Tesseract_Ocr(thresh, psm=13)
                if DEBUG:
                    cv2.imshow("char_croped", char_croped)
                    cv2.imshow("binary_keras", binary_keras)
            else:
                if DEBUG:
                    cv2.imshow("char_croped Fail", char_croped)
                    cv2.imshow("binary_keras Fail", binary_keras)
            if DEBUG:
                cv2.waitKey(0)
    # if DEBUG:
    # cv2.imshow("img_line", img_line)
    # cv2.waitKey(0)
    if DEBUG:
        print(f"char results: {results}")

    return img_line, count_pic_char, LP_result


def get_chars_from_yolo(imgs_LicenceP_Gray, input_size: list, results: list, offset: list = [0, 0], channel=3,
                        DEBUG=False):
    chars_all = []
    binarys_keras = []
    chars_std = []
    for n, result in enumerate(results):
        if result['confidence'] > yolo_confidence:
            char_croped = imgs_LicenceP_Gray.copy()
            if len(char_croped.shape) >= 3:
                char_croped = cv2.cvtColor(char_croped, cv2.COLOR_BGR2GRAY)

            point1 = (result['topleft']['x'], result['topleft']['y'])
            point2 = (result['bottomright']['x'], result['bottomright']['y'])
            croped_height, croped_width = char_croped.shape[:2]

            x1 = point1[0] - offset[0]
            x2 = point2[0] + offset[0]
            y1 = point1[1] - offset[1]
            y2 = point2[1] + offset[1]
            if x1 < 0:
                x1 = 0
            if x2 > croped_width:
                x2 = croped_width

            if y1 < 0:
                y1 = 0
            if y2 > croped_height:
                y2 = croped_height
            # if DEBUG:
            #     print(point1, point2)
            #     print(y1, y2, x1, x2)

            # _char_croped = auto_write_balance(char_croped[point1[1]:point2[1], point1[0]:point2[0]].copy(), 95)
            char_croped = char_croped[y1:y2, x1:x2]

            _char_croped = cv2.GaussianBlur(char_croped, (21, 21), 0)
            char_croped = cv2.GaussianBlur(char_croped, (9, 9), 0)

            print(f'char_croped[{n}]: {np.std(char_croped)}')
            # char_croped = cv2.medianBlur(char_croped, 3)

            _char_croped = auto_write_balance(_char_croped, 999)

            # _char_croped = cv2.fastNlMeansDenoising(_char_croped,
            #                                         h=30,
            #                                         templateWindowSize=7,
            #                                         searchWindowSize=21)

            # _char_croped = histeq(_char_croped, 999)
            # thresh = cv2.adaptiveThreshold(_char_croped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 25)
            # _thresh = cv2.threshold(_char_croped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # thresh = cv2.threshold(_char_croped, 128, 255, cv2.THRESH_BINARY)[1]
            if 1:
                thresh = threshold_li(_char_croped)
                # # thresh = threshold_local(char_croped, block_size, offset=5)
                thresh = np.array(thresh, np.uint8)
                thresh = _char_croped > thresh
                thresh = np.where(thresh == 1, 255, 0)
                thresh = np.array(thresh, np.uint8)
            else:
                thresh = cv2.threshold(_char_croped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            if channel == 3:
                char_croped = cv2.cvtColor(char_croped, cv2.COLOR_GRAY2RGB)
            # else:
            # char_croped = thresh
            if DEBUG:
                # cv2.imshow(f'thresh_{n}', thresh)
                cv2.imshow(f'thresh{n}', thresh)
                print(f'thresh {n} shape: {thresh.shape}')
            # cv2.imshow(f"thresh {n}", thresh)
            if verifyChar(thresh, DEBUG=True) and verifyChar(cv2.bitwise_not(thresh), DEBUG=True):
                # _, _, binary_keras = resize_for_implement_keras(thresh, input_size, [255, 255, 255])

                binary_keras = cv2.resize(char_croped, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)
                # binary_keras = cv2.fastNlMeansDenoising(binary_keras,
                #                                         h=30,
                #                                         templateWindowSize=7,
                #                                         searchWindowSize=21)
                # if len(binary_keras.shape) == 1:
                #     if
                #     binarys_keras.append(cv2.cvtColor(binary_keras, cv2.COLOR_GRAY2RGB))
                # else:
                binarys_keras.append(binary_keras)

                # _, _, char_k = resize_for_keras_model(thresh, input_size, [255, 255, 255], 0)
                # if verifySizes(plate_image, thresh):

                char_k = binary_keras.astype(np.float32)
                char_k = char_k / 255.
                char_k = char_k.reshape(-1, input_size[1], input_size[0], channel)
                chars_all.append(char_k)
    return chars_all, binarys_keras


def get_chars_from_yolov3(imgs_LicenceP_Gray, input_size: list, results: list, offset: list = [0, 0], channel=3,
                          DEBUG=False):
    chars_all = []
    binarys_keras = []
    chars_std = []
    chars_croped = []
    chars_croped_raw = []
    # raw_chars_croped = []
    for n, result in enumerate(results):
        if result['confidence'] > yolo_confidence:
            char_croped = imgs_LicenceP_Gray.copy()
            if len(char_croped.shape) >= 3:
                char_croped = cv2.cvtColor(char_croped, cv2.COLOR_BGR2GRAY)

            point1 = (int(result['topleft']['x']), int(result['topleft']['y']))
            point2 = (int(result['bottomright']['x']), int(result['bottomright']['y']))
            if DEBUG:
                print(point1, point2)
            croped_height, croped_width = char_croped.shape[:2]

            raw_x1 = point1[0]
            raw_x2 = point2[0]
            raw_y1 = point1[1]
            raw_y2 = point2[1]
            if DEBUG:
                print(raw_x1, raw_x2, raw_y1, raw_y2)
            if raw_x1 < 0:
                raw_x1 = 0
            if raw_x2 > croped_width:
                raw_x2 = croped_width

            if raw_y1 < 0:
                raw_y1 = 0
            if raw_y2 > croped_height:
                raw_y2 = croped_height

            x1 = point1[0] - offset[0]
            x2 = point2[0] + offset[0]
            y1 = point1[1] - offset[1]
            y2 = point2[1] + offset[1]
            if x1 < 0:
                x1 = 0
            if x2 > croped_width:
                x2 = croped_width

            if y1 < 0:
                y1 = 0
            if y2 > croped_height:
                y2 = croped_height

            raw_char_croped = char_croped[raw_y1:raw_y2, raw_x1:raw_x2].copy()
            char_croped_raw = imgs_LicenceP_Gray[y1:y2, x1:x2].copy()
            char_croped = char_croped[y1:y2, x1:x2]
            x_sum = cv2.reduce(char_croped, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
            y_sum = cv2.reduce(char_croped, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

            # rotate the vector x_sum
            y_sum = y_sum.transpose()

            # get height and weight
            # y, x = raw_char_croped.shape[:2]
            y, x = char_croped.shape[:2]

            # division the result by height and weight
            x_sum = x_sum / x
            y_sum = y_sum / y

            # convert x_sum to numpy array
            # z = np.array(x_sum)
            # z = np.array(x_sum)
            if DEBUG:
                print(raw_char_croped.shape)
                print(np.std(x_sum), np.std(y_sum))
            raw_char_croped = cv2.GaussianBlur(raw_char_croped, (9, 9), 0)
            chars_std.append(np.std(y_sum) + np.std(x_sum))
            chars_croped_raw.append(char_croped_raw)
            # raw_chars_croped.append(raw_char_croped)

            char_croped = cv2.GaussianBlur(char_croped, (9, 9), 0)

            if DEBUG:
                print(f'raw_char_croped[{n}]: {np.std(y_sum)}')
                print(f'raw_char_croped[{n}]: {np.std(raw_char_croped)}')
            chars_croped.append(char_croped)
    chars_std = np.array(chars_std, dtype=float)

    if DEBUG:
        print(chars_std, chars_std.std(), chars_std.mean(), np.median(chars_std))

    std_median = np.mean(chars_std)
    for n, char_croped in enumerate(chars_croped):
        if channel == 3:
            char_croped = cv2.cvtColor(char_croped, cv2.COLOR_GRAY2RGB)

        # if chars_std[n] > std_median - 15:
        if 1:
            binary_keras = cv2.resize(char_croped, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)

            binarys_keras.append(binary_keras)

            char_k = binary_keras.astype(np.float32)
            char_k = char_k / 255.
            char_k = char_k.reshape(-1, input_size[1], input_size[0], channel)
            chars_all.append(char_k)
    return chars_all, binarys_keras, chars_croped_raw


def get_chars_from_yolov3_result(imgs_LicenceP_Gray, input_size: list, results: list, offset: list = [0, 0], channel=3,
                                 DEBUG=False):
    chars_croped_raw = []
    for n, result in enumerate(results):
        if result['confidence'] > yolo_confidence:
            char_croped = imgs_LicenceP_Gray.copy()
            if len(char_croped.shape) >= 3:
                char_croped = cv2.cvtColor(char_croped, cv2.COLOR_BGR2GRAY)

            point1 = (result['topleft']['x'], result['topleft']['y'])
            point2 = (result['bottomright']['x'], result['bottomright']['y'])
            if DEBUG:
                print(point1, point2)
            croped_height, croped_width = char_croped.shape[:2]

            x1 = point1[0] - offset[0]
            x2 = point2[0] + offset[0]
            y1 = point1[1] - offset[1]
            y2 = point2[1] + offset[1]
            if x1 < 0:
                x1 = 0
            if x2 > croped_width:
                x2 = croped_width

            if y1 < 0:
                y1 = 0
            if y2 > croped_height:
                y2 = croped_height

            chars_croped_raw.append(imgs_LicenceP_Gray[y1:y2, x1:x2])
    return chars_croped_raw


def convert_yolo_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def get_cambodia_section(img, text_detect, input_size=[256, 256]):
    results = text_detect.return_predict(img=img,
                                         index=[0],
                                         input_size=input_size,
                                         equalizeHist=False)
    mask_size = (img.shape[1], img.shape[0])
    _text_area_mask = get_text_mask(results, img
                                    , dest_size=mask_size
                                    , erode_iterations=3
                                    , erode_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                             (1, 7)))
    cv2.imshow("_text_area_mask", _text_area_mask)
    cv2.imshow("text_area_mask", draw_box(img.copy(), results))
    p = get_points_lines(_text_area_mask)
    h_shape, w_shape = _text_area_mask.shape[:2]
    center_w = int(w_shape * 0.37)
    print(center_w)
    t = get_points_text_area(_text_area_mask, text_offset=0, w_min=0)
    start_min = 999
    end_min = 999
    start_m = 0
    end_m = 0
    for _t in t:
        if abs(_t[0] - center_w) < start_min:
            start_min = abs(_t[0] - center_w)
            start_m = _t[0]
        if abs(_t[1] - center_w) < end_min:
            end_min = abs(_t[1] - center_w)
            end_m = _t[1]
    # print(t)
    pv_section = int(min(start_m, end_m) + abs(start_m - end_m) / 2)
    # print(f'start_min: {start_m}')
    # print(f'end_min: {end_m}')
    # print(f'pv_section: {pv_section}')
    pv_result = []
    pv_result_x = []
    for result in results:
        if result['bottomright']['x'] < pv_section:
            pv_result_x.append(result['topleft']['y'])
            pv_result.append(result)

    _, pv_result = zip(*sorted(zip(pv_result_x, pv_result)))
    # print(pv_result)
    # pv_img = img[:, :pv_section, :]
    # cv2.imshow('pv_img', utils.draw_box(pv_img.copy(), pv_result))
    _pv_img = img[:pv_result[0]['bottomright']['y'], :pv_section, :]
    lp_img = img[:, pv_section:, :]

    return lp_img, _pv_img


def overlay_transparent(background, overlay, x, y, alpha: float = 1.0, DEBUG: bool = False):
    if DEBUG:
        print(f"x: {x}, y {y}, alpha: {alpha}")
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = (overlay[..., 3:] / 255.0) * alpha

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def text_group(_results, img, dis_offset=60, char_offset=(0, 0), input_size=(64, 64)):
    img_width = img.shape[1]
    results = reconstruct_charbox(_results, 'v', img_width=img_width, intersect_mix=True)
    # print(_results)
    texts = []
    chars = []
    # offset = offset
    for result in results:
        # print(result)
        if result['label'] == "T":
            texts.append([result, []])
        elif result['label'] == "C":
            chars.append(result)

    # print(chars)

    group_t = []
    t_dict = {}
    for t1, text1 in enumerate(texts):
        for t2, text2 in enumerate(texts):
            if t2 != t1:
                pointt1_y = (0, int(text1[0]['topleft']['y']))
                pointt2_y = (0, int(text2[0]['topleft']['y']))
                dis = distanceBetweenPoints(pointt1_y, pointt2_y)
                if dis < dis_offset:
                    texts[t1][1].append(t2)
    old_text_idx = None
    stack_idx = []
    for t, text in enumerate(texts):
        t_dict[t] = []
        if len(text[1]) == 0:
            for c, char in enumerate(chars):
                point1_t = (0, int(text[0]['topleft']['y']))
                point1_c = (0, int(char['topleft']['y']))
                dis = distanceBetweenPoints(point1_t, point1_c)
                if dis < dis_offset:
                    t_dict[t].append(char)
            # print(f't_dict[{t}]: {t_dict[t]}')
            if len(t_dict[t]) > 0:
                t_dict[t] = reconstruct_charbox(t_dict[t], 'h', img_width=img_width, intersect_mix=True)
            group_t.append((([text[0]]), (tuple(t_dict[t]))))
        else:
            text[1].append(t)
            text[1] = sorted(text[1])

            if old_text_idx != text[1]:
                new_one = True
                if len(stack_idx) > 0:
                    for _stack_idx in stack_idx:
                        if text[1] == _stack_idx:
                            new_one = False
                stack_idx.append(text[1])

                if new_one:
                    txs = []
                    for _tx in text[1]:
                        txs.append(texts[_tx][0])
                    txs = tuple(txs)

                    for c, char in enumerate(chars):
                        point1_t = (0, int(txs[0]['topleft']['y']))
                        point1_c = (0, int(char['topleft']['y']))
                        dis = distanceBetweenPoints(point1_t, point1_c)
                        if dis < dis_offset:
                            t_dict[t].append(char)
                    # print(f't_dict[{t}]: {t_dict[t]}')
                    if len(t_dict[t]) > 0:
                        t_dict[t] = reconstruct_charbox(t_dict[t], 'h', img_width=img_width, intersect_mix=True)
                    group_t.append((txs, (tuple(t_dict[t]))))

            old_text_idx = text[1]
    chars_img = []
    if len(group_t) > 0:
        for i, _group in enumerate(group_t):

            texts, chars = _group
            # print(text)
            # for text in texts:
            # chars_all, binarys_keras, chars_croped_raw = get_chars_from_yolov3(img, input_size, chars, offset=char_offset, DEBUG=False)
            # if len(chars_all) > 0:
            #     chars_img.append((chars_all, binarys_keras, chars_croped_raw))
            chars_img.append(get_chars_from_yolov3(img, input_size, chars, offset=char_offset, DEBUG=False))
            # chars_all, binarys_keras, chars_croped_raw = chars_img[-1]
            # for j, binary_keras in enumerate(binarys_keras):
            #     cv2.imshow(f"chars_img {i} {j}", binary_keras)

    return group_t, chars_img


def draw_rec(img, result_textbox, r, g, b, offset=0):
    for i in range(len(result_textbox)):
        point1 = (result_textbox[i]['topleft']['x'], result_textbox[i]['topleft']['y']+offset)
        point2 = (result_textbox[i]['bottomright']['x'], result_textbox[i]['bottomright']['y']+offset)
        cv2.rectangle(img, point1, point2, (r, g, b), thickness=1)
    return img


def intersection(result_textbox, result_charbox):
    intersec = list();

    for i in range(len(result_textbox)):
        char_index = 0
        char_data_list = list();

        for j in range(len(result_charbox)):

            x11 = result_charbox[j]['topleft']['x']
            x12 = result_charbox[j]['bottomright']['x']
            y11 = result_charbox[j]['topleft']['y']
            y12 = result_charbox[j]['bottomright']['y']
            center_results_x = (x11 + x12) / 2
            center_results_y = (y11 + y12) / 2

            if center_results_x >= result_textbox[i]['topleft']['x'] and center_results_x <= \
                    result_textbox[i]['bottomright']['x']:

                if center_results_y >= result_textbox[i]['topleft']['y'] and center_results_y <= \
                        result_textbox[i]['bottomright']['y']:
                    char_data = {'text_label': result_textbox[i]['label'], 'char_label': result_charbox[j]['label'],
                                 'index_textbox': i, 'index_charbox': j,
                                 'confidence_textbox': result_textbox[i]['confidence'],
                                 'confidence_charbox': result_charbox[j]['confidence'],
                                 'char_topleft': {'x': result_charbox[j]['topleft']['x'],
                                                  'y': result_charbox[j]['topleft']['y']},
                                 'char_bottomright': {'x': result_charbox[j]['bottomright']['x'],
                                                      'y': result_charbox[j]['bottomright']['y']},
                                 'text_topleft': {'x': result_textbox[i]['topleft']['x'],
                                                  'y': result_textbox[i]['topleft']['y']},
                                 'text_bottomright': {'x': result_textbox[i]['bottomright']['x'],
                                                      'y': result_textbox[i]['bottomright']['y']},
                                 'char_center': {'x': center_results_x, 'y': center_results_y},
                                 'char_rotation': 0,
                                 'char_ocr': '!'}
                    char_data_list.append(char_data)

        if len(char_data_list) > 0:
            intersec.append(char_data_list)

    # print('intersec', intersec)
    # print('intersec_lenght', len(intersec))
    # print('charbox_lenght',len(result_charbox))
    # print('textbox_lenght', len(result_textbox))
    return (intersec)


def sequence_check(intersec):
    # print("INTERSEC LIST",intersec)

    CSUM_x1 = 0
    CSUM_x2 = 0
    CSUM_y1 = 0
    CSUM_y2 = 0
    intersec_list = list();

    for i in range(len(intersec)):
        sequence_text = list();
        # print("i=",i)
        # print("intersec data",intersec[i])
        text_H = intersec[i][0]['text_bottomright']['y'] - intersec[i][0]['text_topleft']['y']
        text_V = intersec[i][0]['text_bottomright']['x'] - intersec[i][0]['text_topleft']['x']
        center_v = intersec[i][0]['text_topleft']['x'] + (text_V / 2)
        sequence_index = []
        index_char_min = 0
        rotation = 0
        if text_H < text_V:
            # Vertical  Text
            CSUM_x1 = 0
            CSUM_x2 = 0
            CSUM_y1 = 0
            CSUM_y2 = 0
            for k in range(len(intersec[i])):
                if intersec[i][k]['char_label'] == "SUM":
                    # print('Found CSUM')
                    CSUM_x1 = intersec[i][k]['char_topleft']['x']
                    CSUM_y1 = intersec[i][k]['char_topleft']['y']
                    CSUM_x2 = intersec[i][k]['char_bottomright']['x']
                    CSUM_y2 = intersec[i][k]['char_bottomright']['y']

                    if intersec[i][k]['char_center']['x'] > center_v:
                        # print("Normal rotation")
                        rotation = 0
                    else:
                        # print("180 degree rotation ")
                        rotation = 1

            while len(intersec[i]) > 0:
                # print('loop j ',range(len(intersec[i])))
                char_center_x_min = intersec[i][0]['char_center']['x']
                for k in range(len(intersec[i])):
                    intersec[i][k]['char_rotation'] = rotation
                    # print('loop k ', range(len(intersec[i])), k)
                    if char_center_x_min >= intersec[i][k]['char_center']['x']:
                        char_center_x_min = intersec[i][k]['char_center']['x']
                        index_char_min = k
                    # print('Min', char_center_x_min,"k",k,"index_char_min",index_char_min,"char_center",intersec[i][k]['char_center']['x'],"i",i)
                if (intersec[i][index_char_min]['char_center']['x'] > CSUM_x2 or
                    intersec[i][index_char_min]['char_center']['x'] < CSUM_x1) or intersec[i][index_char_min][
                    'char_label'] == "SUM":

                    sequence_text.append(intersec[i].pop(index_char_min))
                else:
                    # print("CVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
                    temp = intersec[i].pop(index_char_min)

        else:

            CSUM_x1 = 0
            CSUM_x2 = 0
            CSUM_y1 = 0
            CSUM_y2 = 0

            for k in range(len(intersec[i])):
                if intersec[i][k]['char_label'] == "SUM":
                    # print('Found CSUM')
                    CSUM_x1 = intersec[i][k]['char_topleft']['x']
                    CSUM_y1 = intersec[i][k]['char_topleft']['y']
                    CSUM_x2 = intersec[i][k]['char_bottomright']['x']
                    CSUM_y2 = intersec[i][k]['char_bottomright']['y']

            while len(intersec[i]) > 0:
                # print('loop j ',range(len(intersec[i])))
                char_center_y_min = intersec[i][0]['char_center']['y']
                for k in range(len(intersec[i])):
                    intersec[i][k]['char_rotation'] = rotation
                    # print('loop k ', range(len(intersec[i])), k)
                    if char_center_y_min >= intersec[i][k]['char_center']['y']:
                        char_center_y_min = intersec[i][k]['char_center']['y']
                        index_char_min = k
                    # print('Min', char_center_y_min,"k",k,"index_char_min",index_char_min,"char_center",intersec[i][k]['char_center']['y'],"i",i)
                if (intersec[i][index_char_min]['char_center']['y'] > CSUM_y2 or
                    intersec[i][index_char_min]['char_center']['y'] < CSUM_y1) or intersec[i][index_char_min][
                    'char_label'] == "SUM":

                    sequence_text.append(intersec[i].pop(index_char_min))
                else:
                    print("CHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
                    temp = intersec[i].pop(index_char_min)

        intersec_list.append(sequence_text)

    return intersec_list


def number_avg(result):
    number_sort = list();
    number_sum = list();

    for i in range(len(result)):
        print(result[i]["NUMBER"], result[i]['CH'])
        number_sort.append(result[i]["NUMBER"])
    while len(number_sort) > 0:
        number_f = dict();
        number_f["NUMBER"] = number_sort[0]
        number_f["COUNT"] = number_sort.count(number_sort[0])
        number_sum.append(number_f)
        for i in range(number_f["COUNT"]):
            number_sort.remove(number_f["NUMBER"])

    print(number_sum)


class dict2attr(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(dict2attr, self).__setattr__(name, value)
        super(dict2attr, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(dict2attr, self).pop(k, d)


def draw_boxes(img, output, offset=(0, 0)):
    from object_tracker.utils.draw import compute_color_for_labels
    croped = {}
    img_overlay = img.copy()
    img_crop = img.copy()
    for i, box in enumerate(output):
        x1, y1, x2, y2, identity = [int(ii) for ii in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        color = compute_color_for_labels(identity)
        label = '{}{:d}'.format("", identity)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        croped['{}{:d}'.format("", identity)] = img_crop[y1:y2, x1:x2, :]
        # croped.append(img[y1:y2, x1:x2, :].copy())
        # (x1, y1), (x2, y2)
        # if line_cross[1][1] <= y2 <= end_line[1][1]:
        cv2.circle(img_overlay, (x1 + int((x2 - x1) / 2), y2), 5, (0, 0, 255), 5)
        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img_overlay, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img_overlay, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    # cv2.line(img_overlay, line_cross[0], line_cross[1], (255, 0, 0), 2)
    # cv2.line(img_overlay, end_line[0], end_line[1], (0, 0, 255), 2)
    return img_overlay, croped


def _xywh_to_xyxy(im_width, im_height, bbox_xywh):
    x, y, w, h = bbox_xywh
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), im_width - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), im_height - 1)
    return x1, y1, x2, y2

