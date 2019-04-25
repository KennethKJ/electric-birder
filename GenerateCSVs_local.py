
import os
import numpy as np
# Define paths
dB_path = "D:/ML/Databases/Birds_dB/Images/"
bucket_path = "/electric-birder-71281-bird-db/Birds_dB/Images/"
bucket_path = dB_path

# Files

file_list = []
label_list = []

f = open('D:/ML/Databases/Birds_dB/Mappings/classes.txt', 'w')

num_classes = 0
with os.scandir(dB_path) as folders:
    for folder in folders:
        f.write(folder.name + "\n")  # Give your csv text here.

        num_classes += 1
        img_path = dB_path + "/" + folder.name
        with os.scandir(img_path) as files:
            for file in files:
                line = bucket_path + folder.name + "/" + file.name
                file_list.append(line)
                label_list.append(folder.name)
f.close()

print("Total classes = " + str(num_classes))
total_images = len(file_list)
idx = np.random.permutation(total_images)

train_proportion = 0.70
eval_proprotion = 0.2
test_proportion = 1 - train_proportion - eval_proprotion

idx_train = idx[0: int(total_images*train_proportion)]
idx_eval = idx[int(total_images*train_proportion): int(total_images*(train_proportion+eval_proprotion))]
idx_test = idx[int(total_images*(train_proportion+eval_proprotion)): total_images-1]

idxs = [idx_train, idx_eval, idx_test]
tot = len(idx_eval) + len(idx_test) + len(idx_train)

names = ['train', 'eval', 'test']

for j in range(3):

    f = open(names[j] + '_set_local.csv', 'w')
    for i in idxs[j]:
        line = file_list[i] + "," + label_list[i] + "\n"
        print(line)
        f.write(line)  # Give your csv text here.
    f.close()





print("The End")
