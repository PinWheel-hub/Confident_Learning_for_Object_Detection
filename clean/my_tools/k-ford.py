if __name__ == '__main__':
    with open("/data/data_wbw/data/tyre/my_train.txt", "r") as f:       
        train_list = f.read().splitlines()
        train_len = len(train_list)
    k = 5
    for i in range(0, k):
        with open("/data/data_wbw/data/tyre/my_train_{0}.txt".format(i + 1), "w") as f1, open("/data/data_wbw/data/tyre/my_val_{0}.txt".format(i + 1), "w") as f2:
            for j, file in enumerate(train_list):
                if j >= i * train_len / k and j < (i + 1) * train_len / k:
                    f2.write(file + '\n')
                else:
                    f1.write(file + '\n')
