# import os
# import pandas as pd
#
# # 指定txt文件所在的路径
# path = "D:/Sheng/sheng/Xlab/2023/antigen-antibody/3种混合被测物"
#
# # 定义一个空的DataFrame，用于保存所有txt文件的数据
# all_data = pd.DataFrame()
#
# # 遍历路径下的所有txt文件
# for file_name in os.listdir(path):
#     if file_name.endswith(".txt"):
#         # 读取txt文件，并将列名设置为文件名（不包含扩展名）
#         file_path = os.path.join(path, file_name)
#         df = pd.read_csv(file_path, sep="\t", header=None, names=[os.path.splitext(file_name)[0] + "_x", os.path.splitext(file_name)[0] + "_y"])
#
#         # 将当前文件的数据添加到总的DataFrame中
#         all_data = pd.concat([all_data, df], axis=1)
#
# # 将所有数据保存到同一个Excel文件中
# all_data.to_excel("output.xlsx", index=False)



import os
import pandas as pd

# 指定txt文件所在的根路径
root_path = "D:/Sheng/sheng/Xlab/2023/antigen-antibody/单种被测物"

# 定义一个空的字典，用于保存每个子文件夹中的数据
data_dict = {}

# 遍历根路径下的所有子文件夹
for dir_name in os.listdir(root_path):
    dir_path = os.path.join(root_path, dir_name)
    if os.path.isdir(dir_path):
        # 定义一个空的DataFrame，用于保存当前子文件夹中的所有txt数据
        dir_data = pd.DataFrame()

        # 遍历当前子文件夹中的所有txt文件
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".txt"):
                # 读取txt文件，并将列名设置为文件名（不包含扩展名）
                file_path = os.path.join(dir_path, file_name)
                df = pd.read_csv(file_path, sep="\t", header=None, names=[os.path.splitext(file_name)[0] + "_x", os.path.splitext(file_name)[0] + "_y"])

                # 将当前txt的数据添加到当前子文件夹的DataFrame中
                dir_data = pd.concat([dir_data, df], axis=1)

        # 将当前子文件夹的DataFrame添加到字典中
        data_dict[dir_name] = dir_data

# 将所有数据保存到同一个Excel文件中，每个子文件夹的数据保存到一个sheet页中
with pd.ExcelWriter("output.xlsx") as writer:
    for sheet_name, data in data_dict.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)