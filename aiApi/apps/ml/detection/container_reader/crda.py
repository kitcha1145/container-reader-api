import cv2
import os
import numpy as np
import json
import time
import threading
import traceback
from contrainer_utils import Contrainer_model


class PortA:
    def __init__(self, IMG_SHOW=False):
        self.__version__ = "1.2.0"
        self.l = threading.Lock()
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/protocal.json', 'r') as fd:
            self.response_json = json.loads(fd.read())
        self.response_json["version"] = self.__version__
        self.container = Contrainer_model()

    def predict(self, img, mode: int = 0, credit_cost=0, credit_remain=0, debug=False, IMG_SHOW=False):
        try:
            with self.l:
                if isinstance(mode, str):
                    if mode.isdigit():
                        mode = int(mode)
                    else:
                        mode = 0
                snaptime = time.time()
                IN_CONTAINER_q = []

                self.response_json["results"] = []
                self.response_json["processing_time"] = []
                plateoverlay = None
                container_reader_t = time.time()
                vcap_raw, IN_CONTAINER_q = self.container.CONTAINER_READ(img, "", snaptime, IN_CONTAINER_q)

                # print(IN_CONTAINER_q)

                if debug:
                    if len(IN_CONTAINER_q) > 0:
                        _, img_buf = cv2.imencode(".jpg", vcap_raw)
                        return img_buf.tobytes()
                    else:
                        return {"message": "not found"}
                else:
                    if len(IN_CONTAINER_q) > 0:
                        for conn in IN_CONTAINER_q:
                            self.response_json["results"].append(
                                {
                                    "number": conn["number"],
                                    "concheck_result": conn["concheck_result"],
                                    # "time": conn["time"]
                                }
                            )
                            self.response_json["processing_time"].append({
                                "container_reader": time.time() - container_reader_t
                            })
                        return {"message": self.response_json}
                    else:
                        return {"message": "not found"}
        except Exception as err:
            print(traceback.format_exc())
            return {"error": err}

if __name__ == "__main__":
    image = cv2.imread("image/2007.jpg")
    a = PortA()
    res = a.predict(image, 0)
    print(json.dumps(res))
    cv2.imshow("image", image)
    cv2.waitKey(0)