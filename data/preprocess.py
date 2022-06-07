head_string = "label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"

head_string = head_string.replace(",", "\t")

f_train1 = open("train1.txt", encoding="utf-8", mode="r")
f_train2 = open("train2.txt", encoding="utf-8", mode="r")
f_train3 = open("train3.txt", encoding="utf-8", mode="r")


f_train_write = open("train.tsv", encoding="utf-8", mode="w")

f_train_write.write(head_string + "\n")
for line in f_train1:
    f_train_write.write(line)
for line in f_train2:
    f_train_write.write(line)
for line in f_train3:
    f_train_write.write(line)
