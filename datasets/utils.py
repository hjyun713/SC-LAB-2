# import numpy as np
#
#
# def row_processor(row):
#     """
#     :param row:
#     :return: label: [gt] data: [0:3]
#     """
#     label = np.array(row[0:-1], np.float32)  # 64
#     labels = np.reshape(np.eye(4)[label.astype(np.int32)], (10, 4))  # time_slot, cls -> one-hot encoding
#     # return {"label": labels.astype(np.float32), "data": np.array(row[1], dtype=np.float32)}
#
#     # 데이터를 (time_slot, 4) 형식으로 reshape
#     data = np.array(row[-1], dtype=np.float32)  # .T를 사용하여 (4, time_slot) -> (time_slot, 4)
#
#     # 반환 형태는 dictionary
#     return {"label": labels.astype(np.float32), "data": data}
#
# # def row_processor(row):
# #     return {"label": np.array(row[0], dtype=np.int32), "data": np.array(row[1], dtype=np.float32)}
#
#
# def filter_for_data(filename):
#     return filename.endswith(".csv")

import numpy as np

# 불러온 데이터를 타임슬롯 or 타입스텝 단위로 자르기
def row_processor(row):
    """
    :param row:
    :return: label: [gt] data: [0:3]
    """
    label = np.array(row[0], np.float32)  # 64
    # print(label)
    labels = np.reshape(np.eye(4)[label.astype(np.int32)], (64, 4))  # time_slot, cls -> one-hot encoding
    # print(labels.astype(np.float32))
    # print(np.array(row[1]).squeeze())
    return {"label": labels.astype(np.float32), "data": np.array(row[1], dtype=np.float32).squeeze()}
    # return {"label": labels.astype(np.float32), "data": np.array(row[1], dtype=np.float32)}

# def row_processor(row):
#     return {"label": np.array(row[0], dtype=np.int32), "data": np.array(row[1], dtype=np.float32)}


def filter_for_data(filename):
    return filename.endswith(".csv")


