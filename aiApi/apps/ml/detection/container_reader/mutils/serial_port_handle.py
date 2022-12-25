import serial
import serial.tools.list_ports as port
import pynmea2
import multiprocessing


class Serial_handle:
    def __init__(self, baudrate=9600, timeout=5):
        self.detected = False
        self.baudrate = baudrate
        self.timeout = timeout
        self.pt = None
        self.multithread_manager=multiprocessing.Manager()
        self.autorun_destroy=self.multithread_manager.Value(bool, True)

        self.sme_info = {
            'sme_timestamp': "",
            'gps_timestamp': "",
            'hdop': 999.,
            'nsat': 99,
            'speed': 0,
            'gps_status': 99,
            'altitude': -1.,
            'latitude': 0.,
            'longitude': 0.,
            'gsm_version': "",
            'mcu_version': "",
            'ccid': "",
            'status': 0,
            'rvalue': 0
        }
        portlist = port.comports()
        # for p in portlist:
        #   print(p)
        # enumerate
        # print(f'portlist: {portlist}')
        if len(portlist) > 0:
            self.device = portlist[0].device
            print(portlist[0].device)
            self.detected = True

    def start(self, autorun=False, autorun_f=None):
        res = -1
        if self.detected:
            self.pt = serial.Serial(self.device)
            self.pt.baudrate = self.baudrate
            self.pt.timeout = self.timeout
            print(self.pt.getSettingsDict())
            res = 1
            if autorun and autorun_f is not None:
                self.autorun_destroy.set(False)
                sme_autorun = multiprocessing.Process(target=autorun_f, args=(self, self.autorun_destroy))
                sme_autorun.start()
        else:
            print("Can't detect device")
        return res

    def read(self, buffer: int = None):
        if self.pt is not None and self.pt.isOpen():
            if buffer is not None:
                return self.pt.read(buffer)
            else:
                return self.pt.read()
        else:
            print("device is not open")
            return None

    def isOpen(self):
        # print("isOpen")
        if self.detected and self.pt is not None:
            return self.pt.isOpen()
        else:
            return False

    def readline(self, buffer: int = None):
        if self.pt is not None and self.pt.isOpen():
            if buffer is not None:
                return self.pt.readline(buffer)
            else:
                return self.pt.readline()
        else:
            print("device is not open")
            return None

    def stop(self):
        if self.detected and self.pt is not None:
            self.pt.close()

    def SME_decode(self):
        # print('SME_decode')
        if self.detected and self.pt is not None:

            def checkSum(msg: str, chk_str: str):
                check = 0
                for c in msg:
                    check = check ^ ord(c)
                return str(hex(check)).replace("0x", "").upper() == chk_str

            try:
                msg = self.readline().decode("utf-8").replace("\r\n", "")
                if len(msg) > 0:
                    if msg.startswith("$GPGGA") and msg.find("*") != -1:
                        # print(f"$GPGGA: {msg} {checkSum(msg[1:msg.find('*')], msg[msg.find('*') + 1:].upper())}")
                        if checkSum(msg[1:msg.find('*')], msg[msg.find('*') + 1:].upper()):
                            msg1 = pynmea2.parse(msg)
                            self.sme_info['hdop'] = msg1.horizontal_dil
                            self.sme_info['nsat'] = msg1.num_sats
                            self.sme_info['altitude'] = msg1.altitude
                    elif msg.startswith("$GPRMC") and msg.find("*") != -1:
                        # print(f"$GPRMC: {msg} {checkSum(msg[1:msg.find('*')], msg[msg.find('*') + 1:].upper())}")
                        if checkSum(msg[1:msg.find('*')], msg[msg.find('*') + 1:].upper()):
                            # print(msg)
                            msg1 = pynmea2.parse(msg)

                            # self._hour = hour
                            # self._minute = minute
                            # self._second = second
                            # self._microsecond = microsecond
                            # self._tzinfo = tzinfo
                            # self._hashcode = -1
                            # self._fold = fold
                            # print(type(msg1.timestamp))
                            self.sme_info['gps_timestamp'] = msg1.timestamp
                            self.sme_info['gps_status'] = msg1.status
                            self.sme_info['latitude'] = msg1.latitude
                            self.sme_info['longitude'] = msg1.longitude
                            self.sme_info['speed'] = msg1.spd_over_grnd
                    elif msg.startswith("#SME") and msg.find("*") != -1:
                        # print(f"#SME: {msg} {checkSum(msg[:msg.find('*')], msg[msg.find('*') + 1:].upper())}")
                        if checkSum(msg[:msg.find('*')], msg[msg.find('*') + 1:].upper()):
                            sme_p = msg.split("*")
                            sme_d = sme_p[0].split(",")
                            self.sme_info['ccid'] = sme_d[1]
                            self.sme_info['status'] = sme_d[2]
                            self.sme_info['rvalue'] = sme_d[3]
                            self.sme_info['sme_timestamp'] = sme_d[6]
                            self.sme_info['gsm_version'] = sme_d[7]
                            self.sme_info['mcu_version'] = sme_d[8]
                            # print(sme_d)
                    sme_pack = {
                        'gps': {
                            'gps_timestamp': self.sme_info['gps_timestamp'],
                            'sme_timestamp': self.sme_info['sme_timestamp'],
                            'gps_status': self.sme_info['gps_status'],
                            'hdop': float(self.sme_info['hdop']),
                            'nsat': int(self.sme_info['nsat']),
                            'speed': float(self.sme_info['speed']),
                            'altitude': float(self.sme_info['altitude']),
                            'latitude': float(self.sme_info['latitude']),
                            'longitude': float(self.sme_info['longitude']),
                        },
                        'sme': {
                            'gsm_version': self.sme_info['gsm_version'],
                            'mcu_version': self.sme_info['mcu_version'],
                            'ccid': self.sme_info['ccid'],
                            'status': int(self.sme_info['status']),
                            'rvalue': int(self.sme_info['rvalue']),
                        }
                    }
                    # print(sme_pack)
                    return sme_pack
                else:
                    return None
            except Exception as err:
                print(f'readline error: {err}')
                self.stop()
                return None
        else:
            return None


def autorun_f(obj: Serial_handle, do_destroy):
    try:
        while obj.isOpen() and not do_destroy.get():
            print(obj.SME_decode())
        print("autorun stoped")
    finally:
        if obj.isOpen():
            obj.stop()

# try:
#     gps_info.start()
#     while gps_info.isOpen():
#         print(gps_info.SME_decode())
#
#         # bytes().startswith()
#         # print(type(msg))
#         # SME_decode(msg.decode("utf-8").replace("\r\n", ""))
#     # print("s2")
#     # line = gps_info.pt.readlines(1024)
#     # print(line)
#     # print(gps_info.readline())
#     # print("s3")
#     # while gps_info.pt.isOpen():
#     #     print("s2")
#     #     print(gps_info.pt.readlines())
#     #     print("s3")
# finally:
#     if gps_info.isOpen():
#         gps_info.stop()