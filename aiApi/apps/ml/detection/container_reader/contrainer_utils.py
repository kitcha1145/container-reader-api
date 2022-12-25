import cv2
import os
import time
import numpy as np
import tools.alpr_utils as apr
import tools.Section2 as concheck



class Contrainer_model:
    def __init__(self):
        self.IMG_REC = False
        self.save_not_not_detect = True
        self.draw_rec_flag = False

        self.input_h = 416
        self.input_w = 416
        self.anchors = [[10., 13.],
                   [16., 30.],
                   [33., 23.],
                   [30., 61.],
                   [62., 45.],
                   [59., 119.],
                   [116., 90.],
                   [156., 198.],
                   [373., 326.]]
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                      "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "None", "None", "None", "None", "None",
                      "None", "None",
                      "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None",
                      "None",
                      "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None",
                      "None",
                      "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None",
                      "None",
                      "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None",
                      "0", "1",
                      "2", "3", "4", "5", "6", "7", "8", "9"]


        self.Concheck = concheck.container(path=f'{os.path.dirname(os.path.realpath(__file__))}/container_number.csv')

        self.keras_char1 = apr.Keras_model(modelpath=f'{os.path.dirname(os.path.realpath(__file__))}/model/1/aa9-eng-contrainer-char-64.h5')
        self.keras_sum1 = apr.Keras_model(modelpath=f'{os.path.dirname(os.path.realpath(__file__))}/model/1/aa5-eng-csum-3-contrainer-char-64.h5')

        self.yolo_conbox = apr.TF_model_c(classes=["text", "conner", "Lock", "SUM", "C"],
                                       new_size=[self.input_w, self.input_h],
                                       anchors=self.anchors,
                                       restore_path=f'{os.path.dirname(os.path.realpath(__file__))}/model/yolov3-contrainer-ct-train-1_50000.ckpt',
                                       letterbox_resize=False)

    def CSUM_CHECK(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = img / 255.
        img = img.reshape((-1, 64, 64, 3))
        
        result = self.keras_sum1.predict(img)
        
        result = result.reshape(-1)
        scores = list(result)
        # print(char_predicts.shape)
        # print(len(labels))
        index = [x for x in range(0, 90)]
        ocr_results, ocr_labels, ocr_indexs = zip(*sorted(zip(scores, self.labels, index)))

        return ocr_labels[89]

    def checksum(self, text):
        print("CHECK SUM DATA OF :", text)

        Total = 0
        for i in range(len(text)):
            Value = text[i]
            if not Value.isdigit():
                if Value == 'A':
                    Value = 10
                if Value == 'B':
                    Value = 12
                if Value == 'C':
                    Value = 13
                if Value == 'D':
                    Value = 14
                if Value == 'E':
                    Value = 15
                if Value == 'F':
                    Value = 16
                if Value == 'G':
                    Value = 17
                if Value == 'H':
                    Value = 18
                if Value == 'I':
                    Value = 19
                if Value == 'J':
                    Value = 20
                if Value == 'K':
                    Value = 21
                if Value == 'L':
                    Value = 23
                if Value == 'M':
                    Value = 24
                if Value == 'N':
                    Value = 25
                if Value == 'O':
                    Value = 26
                if Value == 'P':
                    Value = 27
                if Value == 'Q':
                    Value = 28
                if Value == 'R':
                    Value = 29
                if Value == 'S':
                    Value = 30
                if Value == 'T':
                    Value = 31
                if Value == 'U':
                    Value = 32
                if Value == 'V':
                    Value = 34
                if Value == 'W':
                    Value = 35
                if Value == 'X':
                    Value = 36
                if Value == 'Y':
                    Value = 37
                if Value == 'Z':
                    Value = 38

            Total += int(Value) * 2 ** i
            TotalA = Total
            TotalB = int(TotalA / 11) * 11
            CSUM = TotalA - TotalB
            if CSUM == 10:
                CSUM = 0
        # print("Check Digit:", str(CSUM))
        if text[3] == 'U' or text[3] == 'J' or text[3] == 'Z':
            return (str(CSUM))
        else:
            return (99)

    def OCR_CHECK(self, img, type):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)
        img = img / 255.
        img = img.reshape((-1, 64, 64, 3))

        result = self.keras_char1.predict(img)
        result = result.reshape(-1)
        scores = list(result)
        # print(char_predicts.shape)
        # print(len(labels))
        index = [x for x in range(0, 90)]
        ocr_results, ocr_labels, ocr_indexs = zip(*sorted(zip(scores, self.labels, index)))

        # print('CHAR',ocr_results[89], ocr_labels[89],ocr_results[88], ocr_labels[88],ocr_results[87], ocr_labels[87])
        if type == "num":
            for j in range(89):
                if ocr_labels[89 - j] == '0' or ocr_labels[89 - j] == '1' or ocr_labels[89 - j] == '2' or ocr_labels[
                    89 - j] == '3' or ocr_labels[89 - j] == '4' or ocr_labels[89 - j] == '5' or ocr_labels[
                    89 - j] == '6' or \
                        ocr_labels[89 - j] == '7' or ocr_labels[89 - j] == '8' or ocr_labels[89 - j] == '9':
                    # print("NUM89=",ocr_labels[89])
                    return ocr_labels[89 - j]

        if type == "az":
            for j in range(89):
                if ocr_labels[89 - j] != '0' and ocr_labels[89 - j] != '1' and ocr_labels[89 - j] != '2' and ocr_labels[
                    89 - j] != '3' and ocr_labels[89 - j] != '4' and ocr_labels[89 - j] != '5' and ocr_labels[
                    89 - j] != '6' and ocr_labels[89 - j] != '7' and ocr_labels[89 - j] != '8' and ocr_labels[
                    89 - j] != '9' and ocr_labels[89 - j] != "None":
                    # print("AZ89=", ocr_labels[89])
                    return ocr_labels[89 - j]

        if type == "ci":
            for j in range(89):
                if ocr_labels[89 - j] == 'U' or ocr_labels[89 - j] == 'J' or ocr_labels[89 - j] == 'Z':
                    return ocr_labels[89 - j]

        # print("ALL89=", ocr_labels[89])
        return ocr_labels[89]

    def CON_BOX(self, vcap):
        result_conbox = self.yolo_conbox.return_predict(_img=vcap, channel=3, index=[0, 1, 2, 3, 4], reconstruct=False,
                                                        threshold=0.1, input_size=[self.input_w, self.input_h])
        vcap = apr.draw_rec(vcap, result_conbox, 255, 255, 255)
        return vcap

    def CONTAINER_READ(self, vcap, gate_in, frame_time, IN_CONTAINER_q):
        try:
            conlist = dict()
            raw_img = vcap.copy()

            result_charbox = list()
            result_textbox = list()
            result_conner = list()
            result_lock = list()
            process = time.time()

            # remove cctv time
            vcap_raw = vcap.copy()
            vcap_h, vcap_w = vcap.shape[:2]
            offset = int(vcap_h*10/100.)
            vcap = vcap[offset:-offset, :, :]
            result_all = self.yolo_conbox.return_predict(_img=vcap, channel=3, index=[0, 1, 2, 3, 4], reconstruct=False,
                                                    threshold=0.1,
                                                    input_size=[self.input_w, self.input_h])
                                                    

            x_limit = int(vcap_w*5/100.)
            y_limit = int(vcap_h*5/100.)

            return vcap_raw, IN_CONTAINER_q
        except Exception as err:
            return vcap, IN_CONTAINER_q