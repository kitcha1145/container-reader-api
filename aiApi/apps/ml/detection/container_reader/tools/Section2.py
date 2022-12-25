import pandas as pd


class container:
    def __init__(self,
                 path):  # init คือ constructor (ฟังก์ชันแรกที่อยู่ใน class) ซึ่ง function ที่อยู่ใน class จะต้องมี self
        df = pd.read_csv(path)
        self.Dict = df.to_dict()
        # print(Dict['Code'])

    def Container_number(self, code: str):
        Con_dict = {'Code': None, 'Company': None}
        if len(code) == 4:
            # print(Dict['Code'].keys())
            for i, char in enumerate(self.Dict['Code'].keys()):
                # print(char, Dict['Code'][char])
                if (code.find(self.Dict['Code'][char]) != -1):
                    # print(code, i,self.Dict['Code'][char], self.Dict['Company'][char])
                    Con_dict['Code'] = self.Dict['Code'][char]
                    Con_dict['Company'] = self.Dict['Company'][char]
                    return Con_dict
            return Con_dict

        else:
            return Con_dict

# Con1 = container(path='container_number.csv') #Con1 คือ class container

# code = Con1.Container_number('JOTU')
# print(code)


# C = "ABCDE"
# print(len(C))
# Q = C.find("ABC")
# print(len("ABC"))
# print(Q)


# with open('container_number.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         print(row)
