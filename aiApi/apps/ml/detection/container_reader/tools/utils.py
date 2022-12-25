# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long
import cv2
import numpy as np


class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= 0).all() and (wh2 >= 0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop


def k_resize(r, desired_size, interpolation=cv2.INTER_AREA):
    old_size = r.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    if min(new_size) <= 0:
        return r
    else:
        return cv2.resize(r, (new_size[1], new_size[0]), interpolation=interpolation)


# Reconstruction function from predict value into plate crpoped from image
def reconstruct(I, img, Iresized, Yr, lp_threshold: float, alpha: float=0.5, dist_size: int=256, edge_offset: list=[0, 0]):
    # 4 max-pooling layers, stride = 2
    net_stride: int = 2 ** 4
    # print(f'net_stride: {net_stride}')
    side: float = ((208 + 40) / 2) / net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)
    # print(f'Yr: {Yr}')
    # print(Yr.shape)

    # print(np.unique(Yr[..., 0]))
    # print(np.unique(Yr[..., 1]))
    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]
    # print(f'Probs: {Probs}')
    # print(f'Affines: {Affines}')

    xx, yy = np.where(Probs > lp_threshold)
    # print(xx, yy)
    # CNN input image size 
    WH = getWH(Iresized.shape)
    # print(f'WH: {WH}')
    # print(f'WH: {WH}')
    # output feature map size
    MN: float = WH / net_stride
    # print(f'MN: {MN}')

    vxx = vyy = alpha  # alpha
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    # print(base)
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]
        # print(affine, prob)

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A * base(vxx, vyy))
        pts_frontal = np.array(B * base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    # print(final_labels_frontal)

    # LP size and type
    TLp = []
    TLp1 = []
    Cor = []
    gain_x: float = 1.0
    gain_y: float = 1.0
    if len(final_labels_frontal) > 0:
        out_size, lp_type = (two_lines, 2) if (
                (final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

        if len(final_labels):
            final_labels.sort(key=lambda x: x.prob(), reverse=True)
            skip_plate = False
            for _, label in enumerate(final_labels):
                # print(getWH(I.shape))
                ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
                # print(f'ptsh bf: {ptsh}')
                if edge_offset[0] > 0 or edge_offset[1] > 0:
                    x = ptsh[0]
                    # print(f'x: {x}')
                    if edge_offset[0] > 0:
                        for j, _x in enumerate(x):
                            if _x < (0+edge_offset[0]):
                                skip_plate = True
                                break
                            elif _x > I.shape[1]:
                                ptsh[0][j] = I.shape[1]
                    else:
                        for j, _x in enumerate(x):
                            if _x < (0+edge_offset[0]):
                                ptsh[0][j] = 0
                            elif _x > I.shape[1]:
                                ptsh[0][j] = I.shape[1]

                    y = ptsh[1]
                    if edge_offset[1] > 0:
                        # print(f'y: {y}')
                        for j, _y in enumerate(y):
                            if _y < (0+edge_offset[1]):
                                skip_plate = True
                                break
                            elif _y > I.shape[0]:
                                ptsh[1][j] = I.shape[0]
                    else:
                        # print(f'y: {y}')
                        for j, _y in enumerate(y):
                            if _y < (0+edge_offset[1]):
                                ptsh[1][j] = 0
                            elif _y > I.shape[0]:
                                ptsh[1][j] = I.shape[0]

                else:
                    x = ptsh[0]
                    # print(f'x: {x}')
                    for j, _x in enumerate(x):
                        if _x < 0:
                            ptsh[0][j] = 0
                        elif _x > I.shape[1]:
                            ptsh[0][j] = I.shape[1]
                    y = ptsh[1]
                    # print(f'y: {y}')
                    for j, _y in enumerate(y):
                        if _y < 0:
                            ptsh[1][j] = 0
                        elif _y > I.shape[0]:
                            ptsh[1][j] = I.shape[0]
                if skip_plate:
                    break
                # print(f'ptsh af: {ptsh}')
                # print(f'ptsh: [{ptsh}], {ptsh.shape}')
                # for i in range(ptsh.shape[0]):
                #     if i == 0:
                #         x = ptsh[i]
                #         print(f'x: {x}')
                #         for j, _x in enumerate(x):
                #             if _x < 0:
                #                 ptsh[i][j] = 0
                #             elif _x > I.shape[1]:
                #                 ptsh[i][j] = I.shape[1]
                #     elif i == 1:
                #         y = ptsh[i]
                #         print(f'y: {y}')
                #         for j, _y in enumerate(y):
                #             if _y < 0:
                #                 ptsh[i][j] = 0
                #             elif _y > I.shape[0]:
                #                 ptsh[i][j] = I.shape[0]
                    # ptsh[i] = ptsh[i]

                # print(f'ptsh: [{ptsh}]')
                x_min = min(ptsh[0])
                x_max = max(ptsh[0])

                y_min = min(ptsh[1])
                y_max = max(ptsh[1])
                gain_x = dist_size / (x_max - x_min)
                gain_y = dist_size / (y_max - y_min)
                out_size1 = (int((x_max - x_min) * gain_x), int((y_max - y_min) * gain_y))
                t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
                t_ptsh1 = getRectPts(0, 0, out_size1[0], out_size1[1])
                # print(x_max,x_min,y_max,y_min)

                H = find_T_matrix(ptsh, t_ptsh)
                H1 = find_T_matrix(ptsh, t_ptsh1)
                # print(f't_ptsh: {t_ptsh}')
                # print(f'ptsh: {ptsh}')
                # print(f'H: [{H}]')
                # print(f'H: {H.shape}, out_size: {out_size.shape}')
                # print(f'H: {H.shape}, out_size: {out_size.shape}')
                Ilp = cv2.warpPerspective(img, H, out_size, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                Ilp1 = cv2.warpPerspective(img, H1, out_size1, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # Ilp = cv2.warpPerspective(img, H, out_size, borderMode=cv2.BORDER_REPLICATE)
                TLp.append(Ilp)
                TLp1.append(Ilp1)
                Cor.append(ptsh)
        return final_labels, TLp, lp_type, Cor, TLp1, gain_x, gain_y
    else:
        return final_labels, TLp, 0, Cor, TLp1, gain_x, gain_y


def plate_draw_box(image, cor, thickness=3):
    pts = []
    x_coordinates = cor[0][0]
    y_coordinates = cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])
    print(f'pts: {pts}')
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    vehicle_image = image

    cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    return vehicle_image


import time


def detect_lp(model, I, img, max_dim, lp_threshold, alpha=0.5, dist_size=256, edge_offset: list=[0, 0]):
    start = time.time()

    min_dim_img = min(I.shape[:2])
    # print(I.shape[1::-1])
    factor = float(max_dim) / min_dim_img
    # print(factor)
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()

    Iresized = cv2.resize(I, (w, h), interpolation=cv2.INTER_NEAREST)
    # Iresized = k_resize(I, 512)
    # cv2.imshow("Iresized", Iresized)
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    T = T.astype(np.float32)
    print(T.shape)
    start_pre = time.time()
    Yr = model.predict(T)
    # print(f'detect_lp predict: {time.time()-start_pre}')

    Yr = np.squeeze(Yr)
    # print(Yr.shape)
    start_recon = time.time()
    L, TLp, lp_type, Cor, TLp1, gain_x, gain_y = reconstruct(I, img, Iresized, Yr, lp_threshold, alpha, dist_size, edge_offset)
    # if len(Cor) > 0:
    #     cv2.imshow("full plate", plate_draw_box(img, Cor))
    print(f'detect_lp predict: {time.time()-start_pre}, start_recon: {time.time()-start_recon}, detect_lp total: {time.time()-start}')
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR), TLp, lp_type, Cor, TLp1, gain_x, gain_y

