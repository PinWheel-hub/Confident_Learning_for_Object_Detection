if __name__ == '__main__':
    with open("../my_wrong_b.txt", 'r') as f:
        msgs = [msg.split(" ", 1)[0][:-1] for msg in f.read().splitlines()]
    with open("../my_data/real_wrong.txt", 'r') as r, open("../my_data/not_wrong.txt", 'r') as n:
        real_wrongs = [msg.split(" ", 1)[0][:-1] if msg[-1] != 'g' else msg for msg in r.read().splitlines()]
        not_wrongs = [msg.split(" ", 1)[0][:-1] if msg[-1] != 'g' else msg  for msg in n.read().splitlines()]
    real = 0
    nt = 0
    with open("not_clear.txt", "w") as f:
        for msg in msgs:
            if msg in real_wrongs:
                real += 1
            elif msg in not_wrongs:
                nt += 1
            else:
                f.write(msg + "\n")
    print(real, nt)