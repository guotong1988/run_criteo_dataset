import math
from model_mlp import RecModel
f = open('./data/train.tsv', encoding="utf-8", mode="r")

data_train = []
data_dev = []
column_index_to_max_id = {}
column_index_to_values = {}
column_index_to_size = {}
for row_index, line in enumerate(f):
    if row_index == 0:
        continue
    if row_index % 1000000 == 0:
        print(row_index)
    splits = line.replace("\n", "").split("\t")
    while len(splits) < 40:
        splits.append("")
    tmp_line = []
    tmp_line.append(int(splits[0]))
    for column_index, one_value in enumerate(splits):
        if column_index >= 1 and column_index <= 13:
            tmp_value = one_value
            if tmp_value == "":
                tmp_value = 0
            else:
                tmp_value = int(tmp_value) + 1
            if tmp_value >= 1:
                tmp_value = int(math.log2(tmp_value)*2)
            elif tmp_value < 1:
                tmp_value = 0
            tmp_line.append(tmp_value)
            if column_index + 39 not in column_index_to_max_id:
                column_index_to_max_id[column_index + 39] = 0
            if tmp_value > column_index_to_max_id[column_index + 39]:
                column_index_to_max_id[column_index + 39] = tmp_value            

        if column_index >= 1 and column_index <= 13:
            if one_value == "":
                one_value = 0
            else:
                one_value = int(one_value) + 1
            if one_value >= 1:
                one_value = int(math.log2(one_value))
            elif one_value < 1:
                one_value = 0
            tmp_line.append(one_value)
            if column_index not in column_index_to_max_id:
                column_index_to_max_id[column_index] = 0
            if one_value > column_index_to_max_id[column_index]:
                column_index_to_max_id[column_index] = one_value
    # for column_index, one_value in enumerate(splits):
        if column_index >= 14:
            if column_index not in column_index_to_values:
                column_index_to_values[column_index] = {}
                column_index_to_values[column_index][""] = 0
                column_index_to_size[column_index] = 1
            if one_value not in column_index_to_values[column_index]:
                column_index_to_size[column_index] = column_index_to_size[column_index] + 1
                column_index_to_values[column_index][one_value] = column_index_to_size[column_index]
                one_value2 = column_index_to_size[column_index]
            else:
                one_value2 = column_index_to_values[column_index][one_value]            

            tmp_line.append(int(one_value2))
            if column_index not in column_index_to_max_id:
                column_index_to_max_id[column_index] = 0
            if one_value2 > column_index_to_max_id[column_index]:
                column_index_to_max_id[column_index] = one_value2

    if row_index % 10 == 0:
        data_dev.append(tmp_line)
    else:
        data_train.append(tmp_line)
all_feature_names = ['I' + str(i) for i in range(1, 27)] + ['II' + str(i) for i in range(1, 27)]

max_id_list = []
for column_index in column_index_to_max_id:
    max_id_list.append(column_index_to_max_id[column_index])
    print(column_index, column_index_to_max_id[column_index])

import tensorflow as tf

with tf.variable_scope("", reuse=tf.AUTO_REUSE):
    model_train = RecModel(
        feature_names_list=all_feature_names,
        max_id_list=max_id_list,
        is_training=True)
with tf.variable_scope("", reuse=True):
    model_dev = RecModel(
        feature_names_list=all_feature_names,
        max_id_list=max_id_list,
        is_training=False)

model_train.train(data_train, data_dev, model_dev)

