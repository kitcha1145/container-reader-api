import threading

import queue
import json
import cv2
import time
import numpy as np
import tools.alpr_utils as utils
import traceback

from skimage.measure import compare_ssim
from concurrent.futures import ThreadPoolExecutor
import object_tracker.vehicle_tracker as vt
import requests


class thread_handle:
    def __init__(self, cfg, tracker, detector, alpr, container, API_URI):
        self.cfg = cfg
        self.rescale = 4
        self.tracker = tracker
        self.detector = detector
        self.alpr = alpr
        self.container = container
        self.max_worker = 50
        self.API_URI = API_URI

        self.watchdog = dict()
        self.watchfps = dict()

        self.raw_image = dict()
        self.crop_image = dict()
        self.model_process_image = dict()
        self.alpr_queue = dict()
        self.container_queue = dict()
        self.send_process_queue = queue.Queue(100)
        self.max_queue = 1000
        self.frame_skip = 2
        self.one_truck_time_max = 60  # 50km/hr -> 13.88m/s 1 container 14m ,max 25m/truck 2 second.

        # self.mutex = dict()
        self.results = dict()
        self.img_debug = {}
        if 0:
            self.rtsp_execute = dict()
            for alpr_k, alpr_v in self.cfg.camera.alpr.items():
                self.watchdog[f"alpr_{alpr_k}"] = 0
                self.watchfps[f"alpr_{alpr_k}"] = 0
                self.raw_image[f"alpr_{alpr_k}"] = None
                self.rtsp_execute[f"alpr_{alpr_k}"] = threading.Thread(target=self.rtsp_reader,
                                                                       args=(alpr_v.url, f"alpr_{alpr_k}",))
            for container_k, container_v in self.cfg.camera.container.items():
                self.watchdog[f"container_{container_k}"] = 0
                self.watchfps[f"container_{container_k}"] = 0
                self.raw_image[f"container_{container_k}"] = None
                self.rtsp_execute[f"container_{container_k}"] = threading.Thread(target=self.rtsp_reader,
                                                                                 args=(container_v.url,
                                                                                       f"container_{container_k}",))

            self.rtsp_execute[f"monitor"] = threading.Thread(target=self.monitor)
            for rtsp_k in self.rtsp_execute.keys():
                self.rtsp_execute[rtsp_k].start()
            for rtsp_k in self.rtsp_execute.keys():
                self.rtsp_execute[rtsp_k].join()
        else:
            worker_count = 0
            self.watchdog["alpr_process"] = 0
            self.watchdog["container_process"] = 0
            for gk in self.cfg.camera.gate.keys():
                # self.mutex[gk] = {}
                self.watchdog[gk] = {}
                self.watchfps[gk] = {}
                self.raw_image[gk] = {}
                self.crop_image[gk] = {}

                self.model_process_image[gk] = {}
                self.results[gk] = {
                    "track_id": [],
                    "plate_time": [],
                    "plate_number": [],
                    "plate_score": [],
                    "plate_province_data": [],
                    "plate_province_score": [],
                    "plate_image": [],
                    "plate_rawimage": [],

                    "container_time": [],
                    "container_number": [],
                    "container_frame": [],
                    "container_mu": [],
                    "container_image": [],
                }
                self.alpr_queue[gk] = {}
                self.container_queue[gk] = {}

                worker_count += len(self.cfg.camera.gate[gk].alpr.items())
                worker_count += len(self.cfg.camera.gate[gk].container.items())

            # self.sync = threading.Barrier(worker_count)
            # worker_count = len(self.cfg.camera.gate.alpr.items())
            # worker_count += len(self.cfg.camera.container.items())
            worker_count += 1  # alpr_process
            worker_count += 1  # container_process
            worker_count += 1  # monitor
            worker_count += 1  # send_process

            print(worker_count)
            # exit()
            if worker_count > self.max_worker:
                print(f"can't create worker more than {self.max_worker}")
                exit()
            self.thread = ThreadPoolExecutor(max_workers=self.max_worker)
            for gk in self.cfg.camera.gate.keys():
                for alpr_k, alpr_v in self.cfg.camera.gate[gk].alpr.items():
                    print(alpr_k)
                    # self.mutex[gk][f"alpr_{alpr_k}"] = threading.Lock()
                    self.watchdog[gk][f"alpr_{alpr_k}"] = 0
                    self.watchfps[gk][f"alpr_{alpr_k}"] = 0
                    self.raw_image[gk][f"alpr_{alpr_k}"] = None
                    self.crop_image[gk][f"alpr_{alpr_k}"] = None
                    self.model_process_image[gk][f"alpr_{alpr_k}"] = None
                    # self.results[gk][f"alpr_{alpr_k}"] = {}
                    self.alpr_queue[gk][f"alpr_{alpr_k}"] = queue.Queue(self.max_queue)
                    self.thread.submit(self.rtsp_reader, alpr_v.url, gk, f"alpr_{alpr_k}", alpr_v.p1, alpr_v.p2,
                                       direction=self.cfg.camera.gate[gk].direction)
                for container_k, container_v in self.cfg.camera.gate[gk].container.items():
                    print(container_k)
                    # self.mutex[gk][f"container_{container_k}"] = threading.Lock()
                    self.watchdog[gk][f"container_{container_k}"] = 0
                    self.watchfps[gk][f"container_{container_k}"] = 0
                    self.raw_image[gk][f"container_{container_k}"] = None
                    self.crop_image[gk][f"container_{container_k}"] = None
                    self.model_process_image[gk][f"container_{container_k}"] = None
                    # self.results[gk][f"container_{container_k}"] = {}
                    self.container_queue[gk][f"container_{container_k}"] = queue.Queue(self.max_queue)
                    self.thread.submit(self.rtsp_reader, container_v.url, gk, f"container_{container_k}",
                                       container_v.p1, container_v.p2, direction=self.cfg.camera.gate[gk].direction)

            self.thread.submit(self.monitor)
            self.thread.submit(self.send_process)
            self.thread.submit(self.alpr_process, "alpr_process")
            self.thread.submit(self.container_process, "container_process")

            # self.results
            # self.mutex[f"container"] = threading.Lock()
            # self.mutex[f"detector"] = threading.Lock()
            # self.mutex[f"alpr"] = threading.Lock()

    def monitor(self):
        start = time.time()
        # cv2.namedWindow("monitor", cv2.WINDOW_AUTOSIZE)
        while True:
            try:
                # print("monitor")
                if time.time() - start > 1:
                    print(f"watchdog: {self.watchdog}")
                    print(f"watchfps: {self.watchfps}")
                    print(f'send_process_queue: {self.send_process_queue.qsize()}')
                    for gk in self.cfg.camera.gate.keys():
                        for alpr_k in self.alpr_queue[gk].keys():
                            print(f"alpr_queue[{alpr_k}]: {self.alpr_queue[gk][alpr_k].qsize()}")
                        for container_k in self.container_queue[gk].keys():
                            print(f"container_queue[{container_k}]: {self.container_queue[gk][container_k].qsize()}")
                    # print(self.results)
                    self.watchdog["alpr_process"] += 1
                    self.watchdog["container_process"] += 1
                    for gk in self.cfg.camera.gate.keys():
                        for wd_k in self.watchdog[gk].keys():
                            self.watchdog[gk][wd_k] += 1
                    start = time.time()

                image_monitor = {}
                for gk in self.cfg.camera.gate.keys():
                    # print(self.raw_image[gk].keys())
                    image_monitor[gk] = []
                    image_monitor_buffer = None

                    for img_idx, raw_img_k in enumerate(self.raw_image[gk].keys()):
                        if self.raw_image[gk][raw_img_k] is None and image_monitor_buffer is None:
                            continue
                        if self.raw_image[gk][raw_img_k] is not None:
                            raw_img = self.raw_image[gk][raw_img_k].copy()
                        if raw_img is None and image_monitor_buffer is not None:
                            raw_img = np.zeros_like(image_monitor_buffer)

                        if raw_img is not None:
                            raw_img_res = cv2.resize(raw_img, (
                                int(raw_img.shape[1] / self.rescale), int(raw_img.shape[0] / self.rescale)),
                                                     cv2.INTER_AREA)
                            raw_img_res = cv2.putText(raw_img_res, raw_img_k, (0, 25), cv2.FONT_ITALIC, 1, (255, 0, 0),
                                                      4)
                            if img_idx % 4 == 0 or img_idx == 0:
                                if img_idx == 0:
                                    image_monitor_buffer = raw_img_res
                                else:
                                    if img_idx == 4:
                                        image_monitor[gk] = [image_monitor_buffer]
                                    else:
                                        image_monitor[gk].append(image_monitor_buffer)

                                    image_monitor_buffer = raw_img_res
                            else:
                                if image_monitor_buffer is not None:
                                    image_monitor_buffer = np.hstack((image_monitor_buffer, raw_img_res))
                    if image_monitor_buffer is not None:
                        image_monitor[gk].append(image_monitor_buffer)

                image_show = {}
                for k in image_monitor.keys():
                    image_show[k] = None
                    if len(image_monitor[k]) > 0:
                        for n, _image_monitor in enumerate(image_monitor[k]):
                            if _image_monitor is not None:
                                if image_show[k] is None:
                                    image_show[k] = _image_monitor
                                else:
                                    image_show[k] = np.vstack((image_show[k], _image_monitor))

                for gk in self.cfg.camera.gate.keys():
                    for img_idx, raw_img_k in enumerate(self.raw_image[gk].keys()):
                        if self.model_process_image[gk][raw_img_k] is not None:
                            cv2.imshow(f"[{gk}] {raw_img_k} raw", self.model_process_image[gk][raw_img_k])

                    for img_idx, crop_image_k in enumerate(self.crop_image[gk].keys()):
                        if self.crop_image[gk][crop_image_k] is not None:
                            cv2.imshow(f"[{gk}] {crop_image_k} crop", self.crop_image[gk][crop_image_k])

                for k in image_monitor.keys():
                    if image_show[k] is not None:
                        cv2.imshow(f"[{k}]image_show", image_show[k])
                # print(self.img_debug)
                for k in self.img_debug.keys():
                    if self.img_debug[k] is not None:
                        cv2.imshow(f"{k}", self.img_debug[k])
                cv2.waitKey(1)
            except Exception as err:
                print(f"monitor: {traceback.format_exc()}")

    def alpr_process(self, pname):
        print(pname)
        while True:
            try:
                for gate in self.alpr_queue.keys():
                    for name in self.alpr_queue[gate].keys():
                        self.watchdog[pname] = 0
                        results = None
                        results_province = None
                        while not self.alpr_queue[gate][name].empty():
                            # self.mutex[name].acquire()
                            # try:
                            # "time": time.time(),
                            # "image": img,
                            # "same": False
                            # print(self.alpr_queue.get())

                            alpr_dict = self.alpr_queue[gate][name].get()
                            image = alpr_dict.get("image")
                            snaptime = alpr_dict.get("time")
                            if image is not None:
                                start = time.time()
                                # self.mutex["detector"].acquire()
                                # try:

                                T = []
                                I_raw = []
                                Iresized = []
                                track_id = []
                                self.model_process_image[gate][name] = utils.k_resize(image, 512)
                                img, croped = self.detection(frame=image)
                                # print(croped.keys())
                                # self.model_process_image[gate][name] = img
                                # finally:
                                #     self.mutex["detector"].release()

                                for i, k in enumerate(croped.keys()):
                                    if i == (len(croped.keys()) - 1):
                                        # image_padded, resize_ratio, dw, dh = self.letterbox_resize1(croped[k], 512, 512)
                                        # image_padded, resize_ratio, dw, dh = self.letterbox_resize1(image, 512, 512)
                                        _T, _Iresized, _I_raw = self.preprocess_img(image)
                                        # _T, _Iresized, _I_raw = self.preprocess_img(cv2.resize(croped[k], (512, 512),
                                        #                                                        interpolation=cv2.INTER_AREA))
                                        T.append(_T)
                                        I_raw.append(_I_raw)
                                        Iresized.append(_Iresized)
                                        track_id.append(k)
                                if len(T) > 0 and len(I_raw) > 0 and len(Iresized) > 0:
                                    T_np = np.array(T, dtype=np.float32)

                                    # start = time.time()
                                    # self.mutex["alpr"].acquire()
                                    # try:
                                    images, Lpimg, lp_type, cor, TLp1, gain_x, gain_y = self.postprocess(
                                        self.alpr.plate_detection_model,
                                        T_np, Iresized, I_raw,
                                        lp_threshold=0.8,
                                        alpha=0.68,
                                        dist_size=256,
                                        edge_offset=[10, 30])
                                    # finally:
                                    #     self.mutex["alpr"].release()
                                    # print(f"len(Lpimg): {len(Lpimg)}")
                                    for i, _Lpimg in enumerate(Lpimg):
                                        if len(_Lpimg) > 0:
                                            for j, __Lpimg in enumerate(_Lpimg):
                                                # print(__Lpimg)
                                                #             # cv2.imshow(f"Lpimg {i},{j}", __Lpimg)
                                                #             # cv2.imshow(f"TLp1 {i},{j}", TLp1[i][j])
                                                # self.mutex["alpr"].acquire()
                                                res = []
                                                try:
                                                    res = self.alpr.plate_ocr(I_raw[i], __Lpimg, TLp1[i][j], gain_x[i],
                                                                              gain_y[i],
                                                                              plate_overlay=None,
                                                                              text_overlay={
                                                                                  'ord': None,
                                                                                  'font-size': 30,
                                                                                  'color': (255, 255, 255)
                                                                              })
                                                except Exception as err:
                                                    print(f"plate_ocr error: {traceback.format_exc()}")
                                                # finally:
                                                #     self.mutex["alpr"].release()
                                                # print(res)
                                                for k, _res in enumerate(res):
                                                    if _res['plate_a'][1] > 70:
                                                        print(f"plate_a: {_res['plate']}")
                                                        print(f"plate_a: {_res['plate_a']}")
                                                        print(f"province: {_res['province']}")
                                                        # self.model_process_image[gate][name] = _res['plate_overlay']
                                                        # print(results)
                                                        if results is None:
                                                            # print(f'1new track_id: {track_id[i]}')
                                                            results = dict()
                                                            results_province = dict()
                                                            results_province[track_id[i]] = [{
                                                                "region": _res['province'][0]["region"],
                                                                "score": _res['province'][0]["confidence"],
                                                                "count": 0,
                                                            }]
                                                            results[track_id[i]] = {
                                                                "plate": [{"plate": _res['plate_a'][0],
                                                                           "score": _res['plate_a'][1],
                                                                           "time": snaptime,
                                                                           "image_overlay": _res['plate_overlay'],
                                                                           "image_raw": I_raw[k],
                                                                           "count": 0}
                                                                          ],
                                                                "got_alpr": False
                                                            }
                                                        else:
                                                            if results.get(track_id[i]) is None:
                                                                # print(f'2new track_id: {track_id[i]}')

                                                                results_province[track_id[i]] = [{
                                                                    "region": _res['province'][0]["region"],
                                                                    "score": _res['province'][0]["confidence"],
                                                                    "count": 0,
                                                                }]
                                                                results[track_id[i]] = {
                                                                    "plate": [{"plate": _res['plate_a'][0],
                                                                               "score": _res['plate_a'][1],
                                                                               "time": snaptime,
                                                                               "image_overlay": _res['plate_overlay'],
                                                                               "image_raw": I_raw[k],
                                                                               "count": 0}
                                                                              ],
                                                                    "got_alpr": False
                                                                }
                                                            else:

                                                                #  update province.
                                                                region = _res['province'][0]["region"]
                                                                score = _res['province'][0]["confidence"]

                                                                got_same_province = False
                                                                for rp_idx, rp in enumerate(
                                                                        results_province[track_id[i]]):
                                                                    if region == results_province[track_id[i]][rp_idx][
                                                                        "region"]:
                                                                        got_same_province = True
                                                                        if score > \
                                                                                results_province[track_id[i]][rp_idx][
                                                                                    "score"]:
                                                                            results_province[track_id[i]][rp_idx][
                                                                                "score"] = score
                                                                        results_province[track_id[i]][rp_idx][
                                                                            "count"] += 1
                                                                if not got_same_province:
                                                                    results_province[track_id[i]].append({
                                                                        "region": _res['province'][0]["region"],
                                                                        "score": _res['province'][0]["confidence"],
                                                                        "count": 0,
                                                                    })
                                                                # for rp_idx, rp in enumerate(results_province[track_id[i]]):
                                                                #     if region == results_province[track_id[i]][rp_idx]["region"]:
                                                                #         if score > results_province[track_id[i]][rp_idx]["score"]:
                                                                #             results_province[track_id[i]][rp_idx]["region"] = score
                                                                #         results_province[track_id[i]][rp_idx]["count"] += 1

                                                                #  update plate.
                                                                plate = _res['plate_a'][0]
                                                                score = _res['plate_a'][1]

                                                                got_same_plate = False
                                                                for rp_idx, rp in enumerate(
                                                                        results[track_id[i]]["plate"]):
                                                                    if plate == results[track_id[i]]["plate"][rp_idx][
                                                                        "plate"]:
                                                                        got_same_plate = True
                                                                        if score > \
                                                                                results[track_id[i]]["plate"][rp_idx][
                                                                                    "score"]:
                                                                            # results[track_id[i]]["plate"][rp_idx]["time"] = snaptime
                                                                            results[track_id[i]]["plate"][rp_idx][
                                                                                "score"] = score
                                                                            results[track_id[i]]["plate"][rp_idx][
                                                                                "image_overlay"] = _res['plate_overlay']

                                                                            results[track_id[i]]["plate"][rp_idx][
                                                                                "image_raw"] = I_raw[k]
                                                                        results[track_id[i]]["plate"][rp_idx][
                                                                            "count"] += 1
                                                                if not got_same_plate:
                                                                    results[track_id[i]]["plate"].append(
                                                                        {
                                                                            "plate": plate,
                                                                            "score": score,
                                                                            "time": snaptime,
                                                                            "image_overlay": _res['plate_overlay'],
                                                                            "image_raw": I_raw[k],
                                                                            "count": 0
                                                                        })
                                # print(f"ALPR time1: {time.time() - start}")
                                # plate = results[track_id[i]]["plate"]
                                # score = results[track_id[i]]["score"]
                                # count = results[track_id[i]]["count"] + 1

                                # if _res['plate_a'][1] > score:
                                #     if plate != _res['plate_a'][0]:
                                #         count = 0
                                #         plate = _res['plate_a'][0]
                                #     score = _res['plate_a'][1]
                                #
                                # results[track_id[i]]["plate"] = plate
                                # results[track_id[i]]["score"] = score
                                # results[track_id[i]]["image_overlay"] = _res['plate_overlay']
                                # results[track_id[i]]["count"] = count

                                # results[track_id[i]] = {
                                #     "plate": plate,
                                #     "score": score,
                                #     "time": snaptime,
                                #     "image_overlay": _res['plate_overlay'],
                                #     "count": count
                                # }
                                # print(results)
                                # for rp_idx, rp in enumerate(results[track_id[i]]["plate"]):
                                #     if results[track_id[i]]["plate"][rp_idx]["count"] > 2:
                                #         results[track_id[i]]["got_alpr"] = True
                                #         self.model_process_image[gate][name] = _res['plate_overlay']
                                #
                                #         max_province_count = -1
                                #         max_province_score = 0
                                #         max_province_data = ""
                                #         for rp_idx, rp in enumerate(results_province[track_id[i]]):
                                #             pcount = results_province[track_id[i]][rp_idx]["count"]
                                #             if pcount > max_province_count:
                                #                 max_province_count = pcount
                                #                 max_province_data = \
                                #                     results_province[track_id[i]][rp_idx]["region"]
                                #         if max_province_count == 0:
                                #             for rp_idx, rp in enumerate(results_province[track_id[i]]):
                                #                 pscore = results_province[track_id[i]][rp_idx]["score"]
                                #                 if pscore > max_province_score:
                                #                     max_province_score = pscore
                                #                     max_province_data = \
                                #                         results_province[track_id[i]][rp_idx]["region"]
                                #         #     if region == results_province[track_id[i]][rp_idx][
                                #         #         "region"]:
                                #
                                #         self.results[gate]["track_id"].append(track_id[i])
                                #         self.results[gate]["plate_time"].append(results[track_id[i]]["plate"][rp_idx]['time'])
                                #         self.results[gate]["plate_number"].append(results[track_id[i]]["plate"][rp_idx]['plate'])
                                #         self.results[gate]["plate_score"].append(results[track_id[i]]["plate"][rp_idx]['score'])
                                #         self.results[gate]["plate_province_data"].append(max_province_data)
                                #         self.results[gate]["plate_province_score"].append(max_province_score)
                                #         self.results[gate]["plate_image"].append(results[track_id[i]]["plate"][rp_idx]["image_overlay"])
                                #         #     = {
                                #         #     "track_id": 0,
                                #         #     "plate_time": 0,
                                #         #     "plate_number": "",
                                #         #     "plate_score": 0,
                                #         #     "plate_province_data": 0,
                                #         #     "plate_province_score": 0,
                                #         #     "plate_image": None,
                                #         #
                                #         #     "container_time": [],
                                #         #     "container_number": [],
                                #         #     "container_frame": [],
                                #         #     "container_mu": [],
                                #         #     "container_image": [],
                                #         # }
                                #         # self.results[gate] = {
                                #         #     "track_id": track_id[i],
                                #         #     "time": results[track_id[i]]['time'],
                                #         #     "plate": results[track_id[i]]['plate'],
                                #         #     "score": results[track_id[i]]['score'],
                                #         #     "province_data": max_province_data,
                                #         #     "province_score": max_province_score,
                                #         #     "image": results[track_id[i]]["image_overlay"]
                                #         # }
                                #         print(f"1result id {track_id[i]} {results[track_id[i]]['plate'][rp_idx]['time']} is: {results[track_id[i]]['plate'][rp_idx]['plate']}, {results[track_id[i]]['plate'][rp_idx]['score']}, {max_province_data}, {max_province_score}")

                        if results is not None and results_province is not None:
                            max_count = {}
                            max_score = {}
                            max_score_result = {}
                            max_score_image = {}
                            max_score_rawimage = {}
                            snaptime = {}

                            max_province_count = {}
                            max_province_score = {}
                            max_province_data = {}
                            # print(results)
                            # print("for 1")
                            for idx, k in enumerate(results.keys()):
                                max_count[k] = -1
                                max_score[k] = 0
                                max_score_result[k] = ""
                                max_score_image[k] = None
                                max_score_rawimage[k] = None
                                snaptime[k] = 0

                                max_province_count[k] = -1
                                max_province_score[k] = 0
                                max_province_data[k] = ""

                            # print("for 2")
                            for idx, k in enumerate(results.keys()):
                                if not results[k]["got_alpr"]:
                                    for rp_idx, rp in enumerate(results[k]["plate"]):
                                        print(f'check LP: {results[k]["plate"][rp_idx]["plate"]} {results[k]["plate"][rp_idx]["score"]} {results[k]["plate"][rp_idx]["count"]}')
                                        if results[k]["plate"][rp_idx]["score"] > max_score[k]:  # results[k]["plate"][rp_idx]["count"] > max_count[k] or
                                            max_count[k] = results[k]["plate"][rp_idx]["count"]
                                            max_score[k] = results[k]["plate"][rp_idx]["score"]
                                            max_score_result[k] = results[k]["plate"][rp_idx]["plate"]
                                            snaptime[k] = results[k]["plate"][rp_idx]["time"]
                                            max_score_image[k] = results[k]["plate"][rp_idx]["image_overlay"]
                                            max_score_rawimage[k] = results[k]["plate"][rp_idx]["image_raw"]

                                    # for rp_idx, rp in enumerate(results_province[k]):
                                    #     pcount = results_province[k][rp_idx]["count"]
                                    #     if pcount > max_province_count[k]:
                                    #         max_province_count[k] = pcount
                                    #         max_province_data[k] = results_province[k][rp_idx]["region"]
                                    # if max_province_count[k] == 0:
                                    for rp_idx, rp in enumerate(results_province[k]):
                                        print(f'check PV: {results_province[k][rp_idx]["region"]} {results_province[k][rp_idx]["score"]}')
                                        pscore = results_province[k][rp_idx]["score"]
                                        if pscore > max_province_score[k]:
                                            max_province_score[k] = pscore
                                            max_province_data[k] = results_province[k][rp_idx]["region"]

                            # print("for 3")
                            # for idx, k in enumerate(results.keys()):
                            #     snaptime[k]
                            track_id = [x for x in results.keys()]
                            plate_time = [snaptime[x] for x in results.keys()]
                            plate_number = [max_score_result[x] for x in results.keys()]
                            plate_score = [max_score[x] for x in results.keys()]
                            plate_province_data = [max_province_data[x] for x in results.keys()]
                            plate_province_score = [max_province_score[x] for x in results.keys()]
                            plate_image = [max_score_image[x] for x in results.keys()]
                            plate_rawimage = [max_score_rawimage[x] for x in results.keys()]
                            _plate_time, _track_id, _plate_number, _plate_score, _plate_province_data, _plate_province_score, _plate_image, _plate_rawimage = zip(*sorted(zip(plate_time, track_id, plate_number, plate_score,
                                            plate_province_data, plate_province_score, plate_image, plate_rawimage)))

                            filter_idx = []
                            _plate_time = list(_plate_time)
                            _track_id = list(_track_id)
                            _plate_number = list(_plate_number)
                            _plate_score = list(_plate_score)
                            _plate_province_data = list(_plate_province_data)
                            _plate_province_score = list(_plate_province_score)
                            _plate_image = list(_plate_image)
                            _plate_rawimage = list(_plate_rawimage)

                            for p in range(1, len(_plate_time), 2):
                                if abs(_plate_time[p] - _plate_time[p-1]) <= 5:
                                    filter_idx.append(p)
                            print(f'plate_time: {_plate_time}')
                            print(f'filter_idx: {filter_idx}')
                            for _filter_idx in filter_idx:
                                _plate_time.pop(_filter_idx)
                                _track_id.pop(_filter_idx)
                                _plate_number.pop(_filter_idx)
                                _plate_score.pop(_filter_idx)
                                _plate_province_data.pop(_filter_idx)
                                _plate_province_score.pop(_filter_idx)
                                _plate_image.pop(_filter_idx)
                                _plate_rawimage.pop(_filter_idx)

                            # for idx, k in enumerate(results.keys()):
                            for k in _track_id:
                                if not results[k]["got_alpr"]:
                                    self.results[gate]["track_id"].append(k)
                                    self.results[gate]["plate_time"].append(snaptime[k])
                                    self.results[gate]["plate_number"].append(max_score_result[k])
                                    self.results[gate]["plate_score"].append(max_score[k])
                                    self.results[gate]["plate_province_data"].append(max_province_data[k])
                                    self.results[gate]["plate_province_score"].append(max_province_score[k])
                                    self.results[gate]["plate_image"].append(max_score_image[k])
                                    self.results[gate]["plate_rawimage"].append(max_score_rawimage[k])

                                    print(f"2result id {k} {snaptime[k]} is: {max_score_result[k]}, {max_score[k]}, {max_province_data[k]}, {max_province_score[k]}")
                                    # print(max_score_image)
                                    self.model_process_image[gate][name] = max_score_image[k]
                                    # cv2.imshow(f"plate_image [{k}]", _res['plate_image'])
                            # print("for End")
                        # self.results[gate][name]["snaptime"] = snaptime
                got_data = False
                for gate in self.alpr_queue.keys():
                    for name in self.alpr_queue[gate].keys():
                        if not self.alpr_queue[gate][name].empty():
                            got_data = True
                if not got_data:
                    time.sleep(0.01)
                # finally:
                #     self.mutex[name].release()
            except Exception as err:
                print(f"alpr {pname} error: {traceback.format_exc()}")

    def container_process(self, pname):
        print(pname)
        isalpr_buffer = False
        iscont_buffer = False
        while True:
            try:
                self.watchdog[pname] = 0

                for gate in self.container_queue.keys():
                    # if iscont_buffer:
                    IN_CONTAINER_q = []
                    got_container = True
                    container_data = {}
                    container_score = []
                    container_gate_process_done = False
                    results_store = None
                    while not container_gate_process_done:
                        for name in self.container_queue[gate].keys():
                            while not self.container_queue[gate][name].empty():
                                # "time": time.time(),
                                # "image": img,
                                # "same": False
                                container_dict = self.container_queue[gate][name].get()
                                image = container_dict.get("image")
                                snaptime = container_dict.get("time")
                                direction = container_dict.get("direction")
                                if image is not None:
                                    # print("To CONTAINER_READ")
                                    start = time.time()
                                    vcap, IN_CONTAINER_q = self.container.CONTAINER_READ(image, direction, snaptime,
                                                                                         IN_CONTAINER_q)
                                    # print(f"CONTAINER_READ time1: {time.time()-start}")
                                    start = time.time()
                                    for CONT in IN_CONTAINER_q:
                                        print(f'con read frame: {CONT["frame"]}')
                                        print(f'con read time: {CONT["time"]}')
                                        print(f'con read number: {CONT["number"]}')
                                        print(f'con read concheck_result: {CONT["concheck_result"]}')
                                        # if container_data.get(CONT["concheck_result"]['Code']) == None:
                                        #     container_data[CONT["concheck_result"]['Code']] = {
                                        #                 "number": CONT["frame"],
                                        #                 "frame": CONT["frame"],
                                        #                 "img": CONT["img"],
                                        #                 "time": CONT["time"],
                                        #                 "concheck_result": CONT["concheck_result"],
                                        #             }
                                        # else:
                                        # container_data[CONT["concheck_result"]['Code']]
                                        # 1633345087.163039
                                        if CONT["frame"] > 0:
                                            if container_data.get(CONT["number"]) is None:

                                                container_data[CONT["number"]] = {
                                                    "frame": CONT["frame"],
                                                    "img": CONT["img"],
                                                    "time": CONT["time"],
                                                    "concheck_result": CONT["concheck_result"],
                                                }
                                                print("1: ", CONT["number"])
                                                print("1: ", container_data[CONT["number"]]["frame"])
                                                print("1: ", container_data[CONT["number"]]["time"])
                                                print("1: ", container_data[CONT["number"]]["concheck_result"])
                                            else:
                                                if CONT["frame"] > container_data[CONT["number"]]["frame"]:
                                                    container_data[CONT["number"]]["frame"] = CONT["frame"]
                                                    container_data[CONT["number"]]["img"] = CONT["img"]
                                                    # container_data[CONT["number"]]["time"] = CONT["time"]
                                                    container_data[CONT["number"]]["concheck_result"] = CONT[
                                                        "concheck_result"]

                                                    print("2: ", CONT["number"])
                                                    print("2: ", container_data[CONT["number"]]["frame"])
                                                    print("2: ", container_data[CONT["number"]]["time"])
                                                    print("2: ", container_data[CONT["number"]]["concheck_result"])
                                            # got_container = True
                                    # print(IN_CONTAINER_q)
                                    # print(f'CONTAINER_READ {time.time() - start}')
                                    # print(f'IN_CONTAINER_q: {IN_CONTAINER_q}')
                                    self.model_process_image[gate][name] = cv2.resize(vcap, (
                                        int(vcap.shape[1] / self.rescale), int(vcap.shape[0] / self.rescale)),
                                                                                      cv2.INTER_AREA)
                                    # print(f"CONTAINER_READ time2: {time.time() - start}")
                                    # print("End CONTAINER_READ")
                                    # self.results[gate][name]["snaptime"] = snaptime

                            # print(container_data.keys())
                        isalpr_buffer = False
                        iscont_buffer = False
                        for k in self.alpr_queue[gate].keys():
                            if not self.alpr_queue[gate][k].empty():
                                isalpr_buffer = True
                        for k in self.container_queue[gate].keys():
                            if not self.container_queue[gate][k].empty():
                                iscont_buffer = True

                        if not iscont_buffer and not isalpr_buffer:
                            container_gate_process_done = True
                            results_store = self.results[gate].copy()
                            self.results[gate] = {
                                "track_id": [],
                                "plate_time": [],
                                "plate_number": [],
                                "plate_score": [],
                                "plate_province_data": [],
                                "plate_province_score": [],
                                "plate_image": [],
                                "plate_rawimage": [],

                                "container_time": [],
                                "container_number": [],
                                "container_frame": [],
                                "container_mu": [],
                                "container_image": [],
                            }
                            break

                    if results_store is not None:
                        start = time.time()
                        if len(container_data.keys()) > 1:
                            container_data_copy = container_data.copy()
                            numbers = [x for x in container_data_copy.keys()]
                            frames = [container_data_copy[x]["frame"] for x in container_data_copy.keys()]
                            times = [container_data_copy[x]["time"] for x in container_data_copy.keys()]
                            imgs = [container_data_copy[x]["img"] for x in container_data_copy.keys()]
                            concheck_results = [container_data_copy[x]["concheck_result"] for x in container_data_copy.keys()]

                            # print(numbers)
                            # print(frames)
                            # print(times)
                            print(f'len: {len(container_data.keys())}')
                            print(len(frames), frames)
                            print(len(times), times)
                            print(len(numbers), numbers)
                            print(len(imgs))
                            print(len(concheck_results))
                            _frames, _times, _numbers, _imgs, _concheck_results = zip(
                                *sorted(zip(frames, times, numbers, imgs, concheck_results)))

                            print(len(_frames), _frames)
                            print(len(_times), _times)
                            print(len(_numbers), _numbers)
                            print(len(_imgs))
                            print(len(_concheck_results))
                            _frames = list(_frames)
                            _times = list(_times)
                            _numbers = list(_numbers)
                            _imgs = list(_imgs)
                            _concheck_results = list(_concheck_results)

                            _frames.reverse()
                            _times.reverse()
                            _numbers.reverse()
                            _imgs.reverse()
                            _concheck_results.reverse()

                            # reconstruct
                            container_data.clear()
                            for i in range(len(_frames)):
                                container_data[_numbers[i]] = {
                                    "frame": _frames[i],
                                    "img": _imgs[i],
                                    "time": _times[i],
                                    "concheck_result": _concheck_results[i],
                                }
                            del container_data_copy

                        for _container_data in container_data.keys():
                            print(f'number e: {_container_data}')
                            print(f'frame e: {container_data[_container_data]["frame"]}')
                            print(f'time e: {container_data[_container_data]["time"]}')
                            print(f'concheck_result e: {container_data[_container_data]["concheck_result"]}')
                            results_store["container_time"].append(container_data[_container_data]["time"])
                            results_store["container_number"].append(_container_data)
                            results_store["container_frame"].append(container_data[_container_data]["frame"])
                            results_store["container_mu"].append(
                                container_data[_container_data]["concheck_result"])
                            results_store["container_image"].append(container_data[_container_data]["img"])
                        if len(results_store["track_id"]) > 0 or len(results_store["container_time"]) > 0:
                            if self.send_process_queue.full():
                                self.send_process_queue.get()
                            self.send_process_queue.put([results_store.copy(), gate])
                        # print(f"CONTAINER_READ time3: {time.time() - start}")
                        # if False:
                        #     got_full_data = False
                        #
                        #     pop_data = []
                        #     data = {}
                        #     c_groups = self.find_cluster(results_store, offset=self.one_truck_time_max)
                        #     if c_groups is not None:
                        #         res_p = results_store["plate_time"]
                        #         res_c = results_store["container_time"]
                        #         print(c_groups)
                        #         for _c_groups in c_groups.keys():
                        #             debug_img = None
                        #             pt_idx = int(_c_groups)
                        #             print(f'{res_p[pt_idx]}: ')
                        #             if len(c_groups[_c_groups]) == 0:
                        #                 pop_data.append(pt_idx)
                        #             else:
                        #                 data = {
                        #                     "track_id": results_store["track_id"][pt_idx],
                        #                     "plate_time": results_store["plate_time"][pt_idx],
                        #                     "plate_number": results_store["plate_number"][pt_idx],
                        #                     "plate_score": results_store["plate_score"][pt_idx],
                        #                     "plate_province_data": results_store["plate_province_data"][pt_idx],
                        #                     "plate_province_score": results_store["plate_province_score"][pt_idx],
                        #                     "plate_image": results_store["plate_image"][pt_idx],
                        #                     "plate_rawimage": results_store["plate_rawimage"][pt_idx],
                        #
                        #                     "container_time": [],
                        #                     "container_number": [],
                        #                     "container_frame": [],
                        #                     "container_mu": [],
                        #                     "container_image": [],
                        #                 }
                        #                 if debug_img is None:
                        #                     debug_img = data["plate_image"]
                        #                     h, w = data["plate_image"].shape[:2]
                        #
                        #                 # pop_cont = []
                        #                 old_time = 0
                        #                 print(f'c_groups[{_c_groups}]: {c_groups[_c_groups]}')
                        #                 for cont in c_groups[_c_groups]:
                        #                     print("\t\t\t", res_c[cont])
                        #                     ctime = results_store["container_time"][cont]
                        #                     if old_time != ctime:
                        #                         # pop_cont.append(cont)
                        #                         data["container_time"].append(ctime)
                        #                         data["container_number"].append(results_store["container_number"][cont])
                        #                         data["container_frame"].append(results_store["container_frame"][cont])
                        #                         data["container_mu"].append(results_store["container_mu"][cont])
                        #                         data["container_image"].append(results_store["container_image"][cont])
                        #                         cimage = results_store["container_image"][cont]
                        #                         cont_resize = cv2.resize(cimage, (w, h))
                        #                         # print(debug_img.shape, )
                        #                         print(f'pt_idx: {pt_idx}', cimage.shape, cont_resize.shape, debug_img.shape)
                        #                         # print(cont_resize.shape)
                        #                         debug_img = np.hstack((debug_img, cont_resize))
                        #                         got_full_data = True
                        #                         old_time = ctime
                        #
                        #             if debug_img is not None:
                        #                 self.img_debug[f"{gate}_result_{pt_idx}"] = debug_img.copy()
                        #             if got_full_data:
                        #                 res = None
                        #                 try:
                        #                     # print(f"_data_group: {_data_group}")
                        #                     if len(data["container_time"]) == 1:
                        #                         res = self.to_format_http(
                        #                             cam_id=gate,
                        #                             snap_time=data["plate_time"],
                        #                             license_plate=data["plate_number"],
                        #                             license_plate_accurate=data["plate_score"],
                        #                             license_province1=data["plate_province_data"],
                        #                             license_province1_accurate=data["plate_province_score"],
                        #                             license_picture_img=data["plate_image"],
                        #                             truck_picture_img=data["plate_rawimage"],
                        #
                        #                             container_no1=data["container_number"][0],
                        #                             container_picture1_img=data["container_image"][0]
                        #                         )
                        #                     elif len(data["container_time"]) >= 2:
                        #                         res = self.to_format_http(
                        #                             cam_id=gate,
                        #                             snap_time=data["plate_time"],
                        #                             license_plate=data["plate_number"],
                        #                             license_plate_accurate=data["plate_score"],
                        #                             license_province1=data["plate_province_data"],
                        #                             license_province1_accurate=data["plate_province_score"],
                        #                             license_picture_img=data["plate_image"],
                        #                             truck_picture_img=data["plate_rawimage"],
                        #
                        #                             container_no1=data["container_number"][0],
                        #                             container_picture1_img=data["container_image"][0],
                        #
                        #                             container_no2=data["container_number"][1],
                        #                             container_picture2_img=data["container_image"][1]
                        #                         )
                        #                     if res is not None:
                        #                         jres = json.loads(res)
                        #                         with open("data.json", 'w') as fd:
                        #                             # print(res)
                        #                             # print(jres)
                        #                             json.dump(jres, fd)
                        #                             # fd.write(json.loads(res))
                        #                         ret = requests.post(self.API_URI, json=jres, timeout=30)
                        #                         print(
                        #                             f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                        #                         with open("request.log", 'w') as fd:
                        #                             fd.write(
                        #                                 f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                        #                 except Exception as err:
                        #                     print(traceback.format_exc())
                        #     if len(pop_data) > 0:
                        #         print(f'pop_data: {pop_data}')
                        #         for idx in pop_data:
                        #             print(f"pop unused: {idx} ", results_store["track_id"])
                        #             print(f"pop unused: {idx} ", results_store["plate_time"])
                        #             print(f"pop unused: {idx} ", results_store["plate_number"])
                        #             print(f"pop unused: {idx} ", results_store["plate_score"])
                        #             print(f"pop unused: {idx} ", results_store["plate_province_data"])
                        #             print(f"pop unused: {idx} ", results_store["plate_province_score"])
                        #
                        #             res = self.to_format_http(
                        #                 cam_id=gate,
                        #                 snap_time=results_store["plate_time"][idx],
                        #                 license_plate=results_store["plate_number"][idx],
                        #                 license_plate_accurate=results_store["plate_score"][idx],
                        #                 license_province1=results_store["plate_province_data"][idx],
                        #                 license_province1_accurate=results_store["plate_province_score"][idx],
                        #                 license_picture_img=results_store["plate_image"][idx],
                        #                 truck_picture_img=results_store["plate_rawimage"][idx],
                        #             )
                        #             if res is not None:
                        #                 jres = json.loads(res)
                        #                 with open("data.json", 'w') as fd:
                        #                     # print(res)
                        #                     # print(jres)
                        #                     json.dump(jres, fd)
                        #                     # fd.write(json.loads(res))
                        #                 ret = requests.post(self.API_URI, json=jres, timeout=30)
                        #                 print(
                        #                     f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                        #                 with open("request.log", 'w') as fd:
                        #                     fd.write(
                        #                         f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                        #         print(results_store["track_id"], results_store["plate_time"],
                        #               results_store["container_time"])

                    # if False:
                    #     if not isalpr_buffer:
                    #         debug_img = None
                    #         for _container_data in container_data.keys():
                    #             print(f'number e: {_container_data}')
                    #             print(f'frame e: {container_data[_container_data]["frame"]}')
                    #             print(f'time e: {container_data[_container_data]["time"]}')
                    #             print(f'concheck_result e: {container_data[_container_data]["concheck_result"]}')
                    #             self.results[gate]["container_time"].append(container_data[_container_data]["time"])
                    #             self.results[gate]["container_number"].append(_container_data)
                    #             self.results[gate]["container_frame"].append(container_data[_container_data]["frame"])
                    #             self.results[gate]["container_mu"].append(
                    #                 container_data[_container_data]["concheck_result"])
                    #             self.results[gate]["container_image"].append(container_data[_container_data]["img"])
                    #
                    #             # if debug_img is None:
                    #             #     debug_img = container_data[_container_data]["img"]
                    #             # else:
                    #             #     debug_img = np.hstack((debug_img, container_data[_container_data]["img"]))
                    #         # map data
                    #         self.find_cluster(self.results[gate])
                    #         # data_group = []
                    #         got_full_data = False
                    #         pop_data = []
                    #         # gate_results = self.results[gate].copy()
                    #         # plate_len = len(self.results[gate]["plate_time"])
                    #         for pt_idx, ptime in enumerate(self.results[gate]["plate_time"]):
                    #             # if plate_len > 1:
                    #             data = {
                    #                 "track_id": self.results[gate]["track_id"][pt_idx],
                    #                 "plate_time": self.results[gate]["plate_time"][pt_idx],
                    #                 "plate_number": self.results[gate]["plate_number"][pt_idx],
                    #                 "plate_score": self.results[gate]["plate_score"][pt_idx],
                    #                 "plate_province_data": self.results[gate]["plate_province_data"][pt_idx],
                    #                 "plate_province_score": self.results[gate]["plate_province_score"][pt_idx],
                    #                 "plate_image": self.results[gate]["plate_image"][pt_idx],
                    #                 "plate_rawimage": self.results[gate]["plate_rawimage"][pt_idx],
                    #
                    #                 "container_time": [],
                    #                 "container_number": [],
                    #                 "container_frame": [],
                    #                 "container_mu": [],
                    #                 "container_image": [],
                    #             }
                    #             if debug_img is None:
                    #                 debug_img = data["plate_image"]
                    #                 h, w = data["plate_image"].shape[:2]
                    #             # else:
                    #             #     debug_img = np.hstack((debug_img, data_group[-1]["plate_image"]))
                    #
                    #             pop_cont = []
                    #             old_time = 0
                    #             for ct_idx, ctime in enumerate(self.results[gate]["container_time"]):
                    #                 if 0 <= ctime - ptime < self.one_truck_time_max and old_time != ctime:
                    #                     pop_cont.append(ct_idx)
                    #                     data["container_time"].append(ctime)
                    #                     data["container_number"].append(self.results[gate]["container_number"][ct_idx])
                    #                     data["container_frame"].append(self.results[gate]["container_frame"][ct_idx])
                    #                     data["container_mu"].append(self.results[gate]["container_mu"][ct_idx])
                    #                     data["container_image"].append(self.results[gate]["container_image"][ct_idx])
                    #
                    #                     cimage = self.results[gate]["container_image"][ct_idx]
                    #                     # h, w = cimage.shape[:2]
                    #                     # h_d, w_d = debug_img.shape[:2]
                    #                     # gain = float(w_d)/w
                    #                     # h_r, w_r = int(h * gain), int(w * gain)
                    #                     # # print(h, w, h_r, w_r)
                    #                     # print(cimage.shape, debug_img.shape)
                    #                     cont_resize = cv2.resize(cimage, (w, h))
                    #                     # print(debug_img.shape, )
                    #                     print(cimage.shape, cont_resize.shape, debug_img.shape)
                    #                     # print(cont_resize.shape)
                    #                     debug_img = np.hstack((debug_img, cont_resize))
                    #                     got_full_data = True
                    #                     old_time = ctime
                    #             if not got_full_data:
                    #                 if len(pop_cont) > 0:
                    #                     for cont_p in pop_cont:
                    #                         self.results[gate]["container_time"].pop(cont_p)
                    #                         self.results[gate]["container_number"].pop(cont_p)
                    #                         self.results[gate]["container_frame"].pop(cont_p)
                    #                         self.results[gate]["container_mu"].pop(cont_p)
                    #                         self.results[gate]["container_image"].pop(cont_p)
                    #                 if (time.time() - ptime) > 20 and not isalpr_buffer and not iscont_buffer:
                    #                     #  TODO sent once.
                    #                     pop_data.append(pt_idx)
                    #                     continue
                    #             # print(self.results[gate])
                    #             if debug_img is not None:
                    #                 self.img_debug[f"{gate}_result_{pt_idx}"] = debug_img
                    #
                    #             if got_full_data:
                    #                 # print(self.results[gate])
                    #                 if True:
                    #                     # for _data_group in data_group:
                    #                     # "track_id": self.results[gate]["track_id"][pt_idx],
                    #                     # "plate_time": self.results[gate]["plate_time"][pt_idx],
                    #                     # "plate_number": self.results[gate]["plate_number"][pt_idx],
                    #                     # "plate_score": self.results[gate]["plate_score"][pt_idx],
                    #                     # "plate_province_data": self.results[gate]["plate_province_data"][pt_idx],
                    #                     # "plate_province_score": self.results[gate]["plate_province_score"][pt_idx],
                    #                     # "plate_image": self.results[gate]["plate_image"][pt_idx],
                    #
                    #                     res = None
                    #                     try:
                    #                         # print(f"_data_group: {_data_group}")
                    #                         if len(data["container_time"]) == 1:
                    #                             res = self.to_format_http(
                    #                                 cam_id=gate,
                    #                                 snap_time=data["plate_time"],
                    #                                 license_plate=data["plate_number"],
                    #                                 license_plate_accurate=data["plate_score"],
                    #                                 license_province1=data["plate_province_data"],
                    #                                 license_province1_accurate=data["plate_province_score"],
                    #                                 license_picture_img=data["plate_image"],
                    #                                 truck_picture_img=data["plate_rawimage"],
                    #
                    #                                 container_no1=data["container_number"][0],
                    #                                 container_picture1_img=data["container_image"][0]
                    #                             )
                    #                         elif len(data["container_time"]) >= 2:
                    #                             res = self.to_format_http(
                    #                                 cam_id=gate,
                    #                                 snap_time=data["plate_time"],
                    #                                 license_plate=data["plate_number"],
                    #                                 license_plate_accurate=data["plate_score"],
                    #                                 license_province1=data["plate_province_data"],
                    #                                 license_province1_accurate=data["plate_province_score"],
                    #                                 license_picture_img=data["plate_image"],
                    #                                 truck_picture_img=data["plate_rawimage"],
                    #
                    #                                 container_no1=data["container_number"][0],
                    #                                 container_picture1_img=data["container_image"][0],
                    #
                    #                                 container_no2=data["container_number"][1],
                    #                                 container_picture2_img=data["container_image"][1]
                    #                             )
                    #                         if res is not None:
                    #                             jres = json.loads(res)
                    #                             with open("data.json", 'w') as fd:
                    #                                 # print(res)
                    #                                 # print(jres)
                    #                                 json.dump(jres, fd)
                    #                                 # fd.write(json.loads(res))
                    #                             ret = requests.post(self.API_URI, json=jres, timeout=30)
                    #                             print(
                    #                                 f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                    #                             with open("request.log", 'w') as fd:
                    #                                 fd.write(
                    #                                     f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                    #                     except Exception as err:
                    #                         print(traceback.format_exc())
                    #                 self.results[gate] = {
                    #                     "track_id": [],
                    #                     "plate_time": [],
                    #                     "plate_number": [],
                    #                     "plate_score": [],
                    #                     "plate_province_data": [],
                    #                     "plate_province_score": [],
                    #                     "plate_image": [],
                    #                     "plate_rawimage": [],
                    #
                    #                     "container_time": [],
                    #                     "container_number": [],
                    #                     "container_frame": [],
                    #                     "container_mu": [],
                    #                     "container_image": [],
                    #                 }
                    #         if len(self.results[gate]["track_id"]) == 0:
                    #             pop_data = []
                    #         if not got_full_data and len(pop_data) > 0:
                    #             for idx in pop_data:
                    #                 print(f"pop unused: {idx} ", self.results[gate]["track_id"])
                    #                 print(f"pop unused: {idx} ", self.results[gate]["plate_time"])
                    #                 print(f"pop unused: {idx} ", self.results[gate]["plate_number"])
                    #                 print(f"pop unused: {idx} ", self.results[gate]["plate_score"])
                    #                 print(f"pop unused: {idx} ", self.results[gate]["plate_province_data"])
                    #                 print(f"pop unused: {idx} ", self.results[gate]["plate_province_score"])
                    #
                    #                 res = self.to_format_http(
                    #                     cam_id=gate,
                    #                     snap_time=self.results[gate]["plate_time"][idx],
                    #                     license_plate=self.results[gate]["plate_number"][idx],
                    #                     license_plate_accurate=self.results[gate]["plate_score"][idx],
                    #                     license_province1=self.results[gate]["plate_province_data"][idx],
                    #                     license_province1_accurate=self.results[gate]["plate_province_score"][idx],
                    #                     license_picture_img=self.results[gate]["plate_image"][idx],
                    #                     truck_picture_img=self.results[gate]["plate_rawimage"][idx],
                    #                 )
                    #                 if res is not None:
                    #                     jres = json.loads(res)
                    #                     with open("data.json", 'w') as fd:
                    #                         # print(res)
                    #                         # print(jres)
                    #                         json.dump(jres, fd)
                    #                         # fd.write(json.loads(res))
                    #                     ret = requests.post(self.API_URI, json=jres, timeout=30)
                    #                     print(
                    #                         f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                    #                     with open("request.log", 'w') as fd:
                    #                         fd.write(
                    #                             f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                    #                 # print(f"pop unused: {idx} ", self.results[gate]["plate_image"])
                    #
                    #                 # "track_id": [],
                    #                 # "plate_time": [],
                    #                 # "plate_number": [],
                    #                 # "plate_score": [],
                    #                 # "plate_province_data": [],
                    #                 # "plate_province_score": [],
                    #                 # "plate_image": []
                    #                 self.results[gate]["track_id"].pop(idx)
                    #                 self.results[gate]["plate_time"].pop(idx)
                    #                 self.results[gate]["plate_number"].pop(idx)
                    #                 self.results[gate]["plate_score"].pop(idx)
                    #                 self.results[gate]["plate_province_data"].pop(idx)
                    #                 self.results[gate]["plate_province_score"].pop(idx)
                    #                 self.results[gate]["plate_image"].pop(idx)
                    #                 self.results[gate]["plate_rawimage"].pop(idx)
                    # for p_idx, gk in enumerate(self.results[gate]["plate_image"]):
                    #     pimage = self.results[gate]["plate_image"][p_idx]
                    #     h, w = pimage.shape[:2]
                    #     h_d, w_d = debug_img.shape[:2]
                    #     gain = float(h_d)/h
                    #
                    #     h_r, w_r = int(h*gain), int(w*gain)
                    #     # print(h, w, h_r, w_r)
                    #     plate_resize = cv2.resize(self.results[gate]["plate_image"][p_idx], (h_r, w_r))
                    #     debug_img = np.hstack((debug_img, plate_resize))
                    # self.model_process_image[gate][debug_windows_name] = debug_img

                    # for cdo_key in container_data.keys():
                    #     cdo_key
                got_data = False

                for gate in self.container_queue.keys():
                    for name in self.container_queue[gate].keys():
                        if not self.container_queue[gate][name].empty():
                            got_data = True
                if not got_data:
                    time.sleep(0.01)
                # else:
                #     print(container_data.keys())
                # time.sleep(1)
            except Exception as err:
                print(f"contrainer {pname}: {traceback.format_exc()}")

    def send_process(self):
        while True:
            try:
                print('send_process...1')
                while not self.send_process_queue.empty():
                    results_store, gate = self.send_process_queue.get()
                    print('send_process...2')
                    got_full_data = False

                    pop_data = []
                    data = {}
                    c_groups = self.find_cluster(results_store, offset=self.one_truck_time_max)
                    if c_groups is not None:
                        res_p = results_store["plate_time"]
                        res_c = results_store["container_time"]
                        print(c_groups)
                        for _c_groups in c_groups.keys():
                            debug_img = None
                            pt_idx = int(_c_groups)
                            print(f'{res_p[pt_idx]}: ')
                            if len(c_groups[_c_groups]) == 0:
                                pop_data.append(pt_idx)
                            else:
                                data = {
                                    "track_id": results_store["track_id"][pt_idx],
                                    "plate_time": results_store["plate_time"][pt_idx],
                                    "plate_number": results_store["plate_number"][pt_idx],
                                    "plate_score": results_store["plate_score"][pt_idx],
                                    "plate_province_data": results_store["plate_province_data"][pt_idx],
                                    "plate_province_score": results_store["plate_province_score"][pt_idx],
                                    "plate_image": results_store["plate_image"][pt_idx],
                                    "plate_rawimage": results_store["plate_rawimage"][pt_idx],

                                    "container_time": [],
                                    "container_number": [],
                                    "container_frame": [],
                                    "container_mu": [],
                                    "container_image": [],
                                }
                                if debug_img is None:
                                    debug_img = data["plate_image"]
                                    h, w = data["plate_image"].shape[:2]

                                # pop_cont = []
                                old_time = 0
                                print(f'c_groups[{_c_groups}]: {c_groups[_c_groups]}')
                                for cont in c_groups[_c_groups]:
                                    print("\t\t\t", res_c[cont])
                                    ctime = results_store["container_time"][cont]
                                    if old_time != ctime:
                                        # pop_cont.append(cont)
                                        data["container_time"].append(ctime)
                                        data["container_number"].append(results_store["container_number"][cont])
                                        data["container_frame"].append(results_store["container_frame"][cont])
                                        data["container_mu"].append(results_store["container_mu"][cont])
                                        data["container_image"].append(results_store["container_image"][cont])
                                        cimage = results_store["container_image"][cont]
                                        cont_resize = cv2.resize(cimage, (w, h))
                                        # print(debug_img.shape, )
                                        print(f'pt_idx: {pt_idx}', cimage.shape, cont_resize.shape, debug_img.shape)
                                        # print(cont_resize.shape)
                                        debug_img = np.hstack((debug_img, cont_resize))
                                        got_full_data = True
                                        old_time = ctime

                            if debug_img is not None:
                                self.img_debug[f"{gate}_result_{pt_idx}"] = debug_img.copy()
                            if got_full_data:
                                res = None
                                try:
                                    # print(f"_data_group: {_data_group}")
                                    if len(data["container_time"]) == 1:
                                        res = self.to_format_http(
                                            cam_id=gate,
                                            snap_time=data["plate_time"],
                                            license_plate=data["plate_number"],
                                            license_plate_accurate=data["plate_score"],
                                            license_province1=data["plate_province_data"],
                                            license_province1_accurate=data["plate_province_score"],
                                            license_picture_img=data["plate_image"],
                                            truck_picture_img=data["plate_rawimage"],

                                            container_no1=data["container_number"][0],
                                            container_picture1_img=data["container_image"][0]
                                        )
                                    elif len(data["container_time"]) >= 2:
                                        res = self.to_format_http(
                                            cam_id=gate,
                                            snap_time=data["plate_time"],
                                            license_plate=data["plate_number"],
                                            license_plate_accurate=data["plate_score"],
                                            license_province1=data["plate_province_data"],
                                            license_province1_accurate=data["plate_province_score"],
                                            license_picture_img=data["plate_image"],
                                            truck_picture_img=data["plate_rawimage"],

                                            container_no1=data["container_number"][0],
                                            container_picture1_img=data["container_image"][0],

                                            container_no2=data["container_number"][1],
                                            container_picture2_img=data["container_image"][1]
                                        )
                                    if res is not None:
                                        jres = json.loads(res)
                                        with open("data.json", 'w') as fd:
                                            # print(res)
                                            # print(jres)
                                            json.dump(jres, fd)
                                            # fd.write(json.loads(res))
                                        print("sending...")
                                        ret = requests.post(self.API_URI, json=jres, timeout=60)
                                        print(
                                            f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                                        with open("request.log", 'w') as fd:
                                            fd.write(
                                                f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                                except Exception as err:
                                    print(traceback.format_exc())

                    else:
                        debug_img = None
                        data = {
                            "container_time": [],
                            "container_number": [],
                            "container_frame": [],
                            "container_mu": [],
                            "container_image": [],
                        }
                        old_time = 0
                        for cont in range(len(results_store["container_time"])):
                            ctime = results_store["container_time"][cont]
                            if old_time != ctime:
                                # pop_cont.append(cont)
                                data["container_time"].append(ctime)
                                data["container_number"].append(results_store["container_number"][cont])
                                data["container_frame"].append(results_store["container_frame"][cont])
                                data["container_mu"].append(results_store["container_mu"][cont])
                                data["container_image"].append(results_store["container_image"][cont])
                                cimage = results_store["container_image"][cont]
                                cont_resize = cv2.resize(cimage, (480, 270))
                                # print(debug_img.shape, )
                                # print(cont_resize.shape)
                                if debug_img is None:
                                    debug_img = cont_resize
                                else:
                                    debug_img = np.hstack((debug_img, cont_resize))
                                print(f'no plate: ', cimage.shape, cont_resize.shape, debug_img.shape)
                                got_full_data = True
                                old_time = ctime
                        if debug_img is not None:
                            self.img_debug[f"{gate}_no plate"] = debug_img.copy()
                        if got_full_data:
                            res = None
                            try:
                                # print(f"_data_group: {_data_group}")
                                if len(data["container_time"]) == 1:
                                    res = self.to_format_http(
                                        cam_id=gate,
                                        snap_time=data["container_time"][0],
                                        license_plate="",
                                        license_plate_accurate=0,
                                        license_province1="",
                                        license_province1_accurate=0,
                                        # license_picture_img=data["plate_image"],
                                        # truck_picture_img=data["plate_rawimage"],

                                        container_no1=data["container_number"][0],
                                        container_picture1_img=data["container_image"][0]
                                    )
                                elif len(data["container_time"]) >= 2:
                                    res = self.to_format_http(
                                        cam_id=gate,
                                        snap_time=data["container_time"][0],
                                        license_plate="",
                                        license_plate_accurate=0,
                                        license_province1="",
                                        license_province1_accurate=0,
                                        # license_picture_img=data["plate_image"],
                                        # truck_picture_img=data["plate_rawimage"],

                                        container_no1=data["container_number"][0],
                                        container_picture1_img=data["container_image"][0],

                                        container_no2=data["container_number"][1],
                                        container_picture2_img=data["container_image"][1]
                                    )
                                if res is not None:
                                    jres = json.loads(res)
                                    with open("data.json", 'w') as fd:
                                        # print(res)
                                        # print(jres)
                                        json.dump(jres, fd)
                                        # fd.write(json.loads(res))
                                    print("no plate sending...")
                                    ret = requests.post(self.API_URI, json=jres, timeout=60)
                                    print(
                                        f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                                    with open("request.log", 'w') as fd:
                                        fd.write(
                                            f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                            except Exception as err:
                                print(traceback.format_exc())

                    if len(pop_data) > 0:
                        print(f'pop_data: {pop_data}')
                        for idx in pop_data:
                            print(f"pop unused: {idx} ", results_store["track_id"])
                            print(f"pop unused: {idx} ", results_store["plate_time"])
                            print(f"pop unused: {idx} ", results_store["plate_number"])
                            print(f"pop unused: {idx} ", results_store["plate_score"])
                            print(f"pop unused: {idx} ", results_store["plate_province_data"])
                            print(f"pop unused: {idx} ", results_store["plate_province_score"])

                            res = self.to_format_http(
                                cam_id=gate,
                                snap_time=results_store["plate_time"][idx],
                                license_plate=results_store["plate_number"][idx],
                                license_plate_accurate=results_store["plate_score"][idx],
                                license_province1=results_store["plate_province_data"][idx],
                                license_province1_accurate=results_store["plate_province_score"][idx],
                                license_picture_img=results_store["plate_image"][idx],
                                truck_picture_img=results_store["plate_rawimage"][idx],
                            )
                            if res is not None:
                                jres = json.loads(res)
                                with open("data.json", 'w') as fd:
                                    # print(res)
                                    # print(jres)
                                    json.dump(jres, fd)
                                    # fd.write(json.loads(res))
                                ret = requests.post(self.API_URI, json=jres, timeout=60)
                                print(
                                    f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                                with open("request.log", 'w') as fd:
                                    fd.write(
                                        f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}] ret: {ret}, {ret.content}')
                        print(results_store["track_id"], results_store["plate_time"],
                              results_store["container_time"])
            except Exception as err:
                print(f"send_process error: {traceback.format_exc()}")
            time.sleep(0.5)

    def find_cluster(self, gate_results, offset=20):
        # print(gate_results["plate"])
        if len(gate_results["plate_time"]) > 0:
            res_p = gate_results["plate_time"].copy()
            res_c = gate_results["container_time"].copy()
            c_groups = {}
            if len(res_p) > 1:
                for i in range(1, len(res_p), 1):
                    c_group = []
                    for j in range(len(res_c)):
                        if res_p[i - 1] <= res_c[j] < res_p[i]:
                            c_group.append(j)
                    c_groups[f"{i - 1}"] = c_group

                c_group = []
                for j in range(len(res_c)):
                    if res_p[-1] <= res_c[j] < res_p[-1] + offset:
                        c_group.append(j)
                c_groups[f"{len(res_p) - 1}"] = c_group
                del res_p
                del res_c
                return c_groups
            elif len(res_p) == 1:
                c_group = []
                for j in range(len(res_c)):
                    if res_p[0] <= res_c[j] < res_p[0] + offset:
                        c_group.append(j)
                c_groups[f"{0}"] = c_group
                del res_p
                del res_c
                return c_groups
            else:
                del res_p
                del res_c
                return None
        else:
            return None

    def rtsp_reader(self, path, gate, name, p1, p2, direction):
        print(path, gate, name, p1, p2, direction)
        start = time.time()
        cap = cv2.VideoCapture(path)
        # self.sync.wait()
        while True:
            try:
                old_frame = None
                fps = 0
                self.watchdog[gate][name] = 0
                print(f"[{name}]rtsp_reader reset")
                while cap.grab():
                    ret, img = cap.retrieve()
                    # print(ret)
                    if time.time() - start >= 1:
                        self.watchfps[gate][name] = fps
                        fps = 0
                        start = time.time()
                    if ret:
                        # print(img.shape)
                        fps += 1
                        # time.sleep(0.02)
                        gray = cv2.cvtColor(img[p1[1]:p2[1], p1[0]:p2[0], :], cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)

                        if old_frame is not None:
                            (score, diff) = compare_ssim(gray, old_frame, full=True)
                            # print(f"score diff: {score}")

                            if name.find("alpr") == 0:
                                if self.alpr_queue[gate][name].full():
                                    self.alpr_queue[gate][name].get()
                                    # frame_count = 0
                                if score < 0.7:
                                    # if frame_count > self.frame_skip or frame_count == 0 or True:
                                    self.alpr_queue[gate][name].put({
                                        "time": time.time(),
                                        "image": img,
                                        # "image_config": cv2.rectangle(img.copy(), p1, p2, (0, 255, 0), 2),
                                        "same": False
                                    })
                                    #     frame_count = 0
                                    # frame_count += 1
                            elif name.find("container") == 0:
                                if self.container_queue[gate][name].full():
                                    self.container_queue[gate][name].get()
                                if score < 0.7:
                                    # if frame_count > self.frame_skip or frame_count == 0:
                                    self.container_queue[gate][name].put({
                                        "time": time.time(),
                                        "image": img,
                                        "direction": direction,
                                        # "image_config": cv2.rectangle(img.copy(), p1, p2, (0, 255, 0), 2),
                                        "same": False
                                    })
                                    # frame_count = 0
                                    # frame_count += 1
                            # else:
                            #     self.alpr_queue.put({
                            #         "time": time.time(),
                            #         "image": img,
                            #         "same": True
                            #     })
                        # 1633077384.7454875
                        # 1633077383.9813895
                        old_frame = gray
                        # self.mutex[gate][name].acquire()
                        # try:
                        self.raw_image[gate][name] = cv2.rectangle(img=img.copy(), pt1=tuple(p1), pt2=tuple(p2),
                                                                   color=(0, 255, 0), thickness=2)
                        self.crop_image[gate][name] = gray
                        # finally:
                        #     self.mutex[gate][name].release()
                        self.watchdog[gate][name] = 0

                    else:
                        break
                cap.release()
                time.sleep(10)
                cap = cv2.VideoCapture(path)
            except Exception as err:
                print(f"rtsp {gate}, {name} error: {traceback.format_exc()}")

    def detection(self, frame, detection_imshow=False):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        # print(f'bbox_xywh: {bbox_xywh}')
        # print(f'cls_conf: {cls_conf}')
        # print(f'cls_ids: {cls_ids}')
        img, croped = None, {}
        if bbox_xywh is not None:
            # select person class
            # print(f'cls_ids: {cls_ids}')
            mask = np.where(((cls_ids == 2) |
                             (cls_ids == 3) |
                             (cls_ids == 5) |
                             (cls_ids == 7)), True, False)
            # mask1 = cls_ids == 2
            # mask2 = cls_ids == 3
            # mask3 = cls_ids == 5
            # mask4 = cls_ids == 7
            # mask12 = np.logical_or(mask1, mask2)
            # mask123 = np.logical_or(mask12, mask3)
            # mask1234 = np.logical_or(mask123, mask4)

            # print(f'mask1234: {mask1234}')
            bbox_xywh = bbox_xywh[mask]
            # bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
            cls_conf = cls_conf[mask]

            bbox_xyxy = [utils._xywh_to_xyxy(im.shape[1], im.shape[0], x) for x in bbox_xywh]

            # mask1 = [False for i in range(len(bbox_xyxy))]
            # for i, _bbox_xyxy in enumerate(bbox_xyxy):
            #     x1, y1, x2, y2 = _bbox_xyxy
            #     # if self.cross_line_point[1][1] <= y2 <= self.end_line_point[1][1]:
            #     mask1[i] = True

            # print(bbox_xywh.shape)
            # bbox_xywh = bbox_xywh[mask1]
            # bbox_xyxy = np.array(bbox_xyxy)[mask1]
            # cls_conf = cls_conf[mask1]

            if len(bbox_xywh) > 0:
                img1 = frame.copy()
                for i, box in enumerate(bbox_xyxy):
                    x1, y1, x2, y2 = [int(ii) for ii in box]
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    label = '{}{:.2f}'.format("", cls_conf[i])
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(img1, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (255, 255, 255), -1)
                    cv2.putText(img1, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 2)
                if detection_imshow:
                    cv2.imshow("detection", img1)
                # print(bbox_xywh.shape)
                # do tracking
                outputs = self.tracker.update(bbox_xywh, cls_conf, im)
                # if line_cross[1][1] <= y2 <= end_line[1][1]:
                # print(bbox_xyxy)
                # draw boxes for visualization

                img, croped = utils.draw_boxes(frame, outputs)
            else:
                img, croped = utils.draw_boxes(frame, [])

        return img, croped

    @staticmethod
    def preprocess_img(image):
        Dmax = 608
        Dmin = 300

        # _image_r = image
        _image = image.copy()
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _image = cv2.cvtColor(_image, cv2.COLOR_GRAY2RGB)
        _image = _image / 255.

        ratio = float(max(_image.shape[:2]) / min(_image.shape[:2]))
        side = int(ratio * Dmin)
        # print(f'side: {side}')
        bound_dim = min(side, Dmax)
        min_dim_img = min(_image.shape[:2])
        factor = float(bound_dim) / min_dim_img
        w, h = (np.array(_image.shape[1::-1], dtype=float) * factor).astype(int).tolist()
        Iresized = cv2.resize(_image, (w, h), interpolation=cv2.INTER_NEAREST)

        T = Iresized.copy()
        T = T.astype(np.float32)

        # return np.expand_dims(T, axis=0), np.expand_dims(Iresized, axis=0), np.expand_dims(image, axis=0)
        return T, Iresized, image

    @staticmethod
    def postprocess(plate_detect, T, Iresized, I_raw, lp_threshold, alpha=0.5, dist_size=256,
                    edge_offset: list = [0, 0]):
        from tools.utils import reconstruct
        h, w = Iresized[0].shape[:2]
        Yr = plate_detect.predict(T)
        L = []
        TLp = []
        lp_type = []
        Cor = []
        TLp1 = []
        gain_x = []
        gain_y = []
        imgs = []
        for i, _Yr in enumerate(Yr):
            _Yr = np.squeeze(_Yr)
            # print(I_raw[i].shape)
            # print(Iresized[i].shape)
            _L, _TLp, _lp_type, _Cor, _TLp1, _gain_x, _gain_y = reconstruct(I_raw[i], I_raw[i], Iresized[i], _Yr,
                                                                            lp_threshold, alpha, dist_size,
                                                                            edge_offset)

            # print(_TLp)
            L.append(_L)
            TLp.append(_TLp)
            lp_type.append(_lp_type)
            Cor.append(_Cor)
            TLp1.append(_TLp1)
            gain_x.append(_gain_x)
            gain_y.append(_gain_y)
            imgs.append(cv2.resize(I_raw[i], (w, h), interpolation=cv2.INTER_LINEAR))

        return imgs, TLp, lp_type, Cor, TLp1, gain_x, gain_y

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

    @staticmethod
    def to_format_http(cam_id: str,
                       snap_time: float,

                       license_plate: str = "",
                       license_plate_accurate: int = 0,
                       license_province1: str = "",
                       license_province1_accurate: int = 0,
                       # license_province2: str = "",
                       # license_province2_accurate: str = "",
                       license_picture_img=None,
                       truck_picture_img=None,
                       container_no1: str = "",
                       container_picture1_img=None,
                       container_no2: str = "",
                       container_picture2_img=None,
                       ):
        import base64
        # print(cam_id)
        # print(snap_time)
        # print(license_plate)
        # print(license_plate_accurate)
        # print(license_province1)
        # print(license_province1_accurate)
        # print(license_picture_img)
        # print(truck_picture_img)
        # print(container_no1)
        # print(container_picture1_img)
        # print(container_no2)
        # print(container_picture2_img)
        # path="Plate/Plategreen_ศ1199_กรุงเทพมหานคร.jpg"
        # img=cv2.imread(path)
        # img = k_resize(img, 256)
        if license_picture_img is not None:
            _, license_picture_arr = cv2.imencode('.jpg',
                                                  license_picture_img)  # im_arr: image in Numpy one-dim array format.
            license_picture_bytes = license_picture_arr.tobytes()
            license_picture_base64 = base64.b64encode(license_picture_bytes)

        if truck_picture_img is not None:
            _, truck_picture_arr = cv2.imencode('.jpg', truck_picture_img)  # im_arr: image in Numpy one-dim array format.
            truck_picture_bytes = truck_picture_arr.tobytes()
            truck_picture_base64 = base64.b64encode(truck_picture_bytes)

        if container_picture1_img is not None:
            _, container_picture1_arr = cv2.imencode('.jpg',
                                                     container_picture1_img)  # im_arr: image in Numpy one-dim array format.
            container_picture1_bytes = container_picture1_arr.tobytes()
            container_picture1_base64 = base64.b64encode(container_picture1_bytes)

        if container_picture2_img is not None:
            _, container_picture2_arr = cv2.imencode('.jpg',
                                                     container_picture2_img)  # im_arr: image in Numpy one-dim array format.
            container_picture2_bytes = container_picture2_arr.tobytes()
            container_picture2_base64 = base64.b64encode(container_picture2_bytes)

            # res = "{"\
            #       + f'"cam_id" : "{cam_id}",'\
            #       + f'"snap_time" : "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap_time))}",'\
            #       + f'"license_plate" : "{cam_id}",'\
            #       + f'"license_plate_accurate" : "{cam_id}",'\
            #       + f'"license_province1" : "{cam_id}",'\
            #       + f'"license_province1_accurate" : "{cam_id}",'\
            #       + f'"license_province2" : "{cam_id}",'\
            #       + f'"license_province2_accurate" : "{cam_id}",'\
            #       + f'"license_picture" : "{cam_id}",'\
            #       + f'"truck_picture" : "{cam_id}",'\
            #       + f'"container_no1" : "{cam_id}",'\
            #       + f'"container_picture1" : "{cam_id}",'\
            #       + f'"container_no2" : "{cam_id}",'\
            #       + f'"container_picture2" : "data:image/jpeg;base64,{container_picture2_base64.decode("utf-8")}",'\
            #       + "}"
            # res = f''

        res = f'{{"cam_id" : "{cam_id}",\
            "snap_time" : "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap_time))}",\
            "license_plate" : "{license_plate}",\
            "license_plate_accurate" : "{license_plate_accurate}%",\
            "license_province1" : "{license_province1}",\
            "license_province1_accurate" : "{license_province1_accurate}%",\
            "license_province2" : "",\
            "license_province2_accurate" : "0%"'

        if license_picture_img is not None:
            res += f',"license_picture" : "data:image/jpeg;base64,{license_picture_base64.decode("utf-8")}"'

        if truck_picture_img is not None:
            res += f',"truck_picture": "data:image/jpeg;base64,{truck_picture_base64.decode("utf-8")}"'

        if container_picture1_img is not None:
            res += f',"container_no1": "{container_no1}"'
            res += f',"container_picture1": "data:image/jpeg;base64,{container_picture1_base64.decode("utf-8")}"'

        if container_picture2_img is not None:
            res += f',"container_no2": "{container_no2}"'
            res += f',"container_picture2": "data:image/jpeg;base64,{container_picture2_base64.decode("utf-8")}"'
        res += f'}}'
        # if container_picture1_img is not None and container_picture2_img is not None:
        #     res = f'{{"cam_id" : "{cam_id}",\
        #         "snap_time" : "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap_time))}",\
        #         "license_plate" : "{license_plate}",\
        #         "license_plate_accurate" : "{license_plate_accurate}%",\
        #         "license_province1" : "{license_province1}",\
        #         "license_province1_accurate" : "{license_province1_accurate}%",\
        #         "license_province2" : "",\
        #         "license_province2_accurate" : "0%",\
        #         "license_picture" : "data:image/jpeg;base64,{license_picture_base64.decode("utf-8")}",\
        #         "truck_picture": "data:image/jpeg;base64,{truck_picture_base64.decode("utf-8")}",\
        #         "container_no1": "{container_no1}",\
        #         "container_picture1": "data:image/jpeg;base64,{container_picture1_base64.decode("utf-8")}",\
        #         "container_no2": "{container_no2}",\
        #         "container_picture2": "data:image/jpeg;base64,{container_picture2_base64.decode("utf-8")}"\
        #         }}'
        # elif container_picture1_img is not None and container_picture2_img is None:
        #     res = f'{{"cam_id" : "{cam_id}",\
        #         "snap_time" : "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap_time))}",\
        #         "license_plate" : "{license_plate}",\
        #         "license_plate_accurate" : "{license_plate_accurate}%",\
        #         "license_province1" : "{license_province1}",\
        #         "license_province1_accurate" : "{license_province1_accurate}%",\
        #         "license_province2" : "",\
        #         "license_province2_accurate" : "0%",\
        #         "license_picture" : "data:image/jpeg;base64,{license_picture_base64.decode("utf-8")}",\
        #         "truck_picture": "data:image/jpeg;base64,{truck_picture_base64.decode("utf-8")}",\
        #         "container_no1": "{container_no1}",\
        #         "container_picture1": "data:image/jpeg;base64,{container_picture1_base64.decode("utf-8")}"\
        #         }}'
        # elif container_picture1_img is None and container_picture2_img is None:
        #     res = f'{{"cam_id" : "{cam_id}",\
        #         "snap_time" : "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap_time))}",\
        #         "license_plate" : "{license_plate}",\
        #         "license_plate_accurate" : "{license_plate_accurate}%",\
        #         "license_province1" : "{license_province1}",\
        #         "license_province1_accurate" : "{license_province1_accurate}%",\
        #         "license_province2" : "",\
        #         "license_province2_accurate" : "0%",\
        #         "license_picture" : "data:image/jpeg;base64,{license_picture_base64.decode("utf-8")}",\
        #         "truck_picture": "data:image/jpeg;base64,{truck_picture_base64.decode("utf-8")}"\
        #         }}'
        # elif type == 'container':
        #     res = f'{{"type" : "{type}",\
        #         "snap_time" : "{snap_time}",\
        #         "container_number" : "{container_number}",\
        #         "location" : "{location}",\
        #         "picture" : "data:image/jpeg;base64,{im_b64.decode("utf-8")}"}}'

        # print(res)
        return res
