import numpy as np
np.random.seed(1111)

pos_path = '/Users/rizwanparvez/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.pos'
neg_path = '/Users/rizwanparvez/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.neg'


with open(pos_path, 'r') as pf:
    pos_all = [line.replace("\n", "") for line in pf]

with open(neg_path, 'r') as nf:
    neg_all = [line.replace("\n", "") for line in nf]
print(len(pos_all))
print(len(neg_all))

dev_test =  np.random.choice(len(pos_all), 1000)
dev = dev_test[:500]
test = dev_test[500:]
print(len(dev))
print(len(test))

trp = [pos_all[i] for i in range(len(pos_all)) if i not in dev_test]
trn = [neg_all[i] for i in range(len(neg_all)) if i not in dev_test]
devp = [pos_all[i] for i in dev]
devn = [neg_all[i] for i in dev]
testp = [pos_all[i] for i in test]
testn = [neg_all[i] for i in test]


with open('/Users/rizwanparvez/Downloads/rt-polaritydata/rt-polaritydata/train_rotten_tomatoes.txt', 'w') as trf:
    for i in range(len(trp)):
        line = trp[i]
        trf.write('1'+'\t'+line+'\n')
        line = trn[i]
        trf.write('0' + '\t' + line + '\n')

with open('/Users/rizwanparvez/Downloads/rt-polaritydata/rt-polaritydata/dev_rotten_tomatoes.txt', 'w') as trf:
    for i in range(len(devp)):
        line = devp[i]
        trf.write('1' + '\t' + line + '\n')
        line = devn[i]
        trf.write('0' + '\t' + line + '\n')

with open('/Users/rizwanparvez/Downloads/rt-polaritydata/rt-polaritydata/test_rotten_tomatoes.txt', 'w') as trf:
    for i in range(len(testp)):
        line = testp[i]
        trf.write('1' + '\t' + line + '\n')
        line = testn[i]
        trf.write('0' + '\t' + line + '\n')

import os
train_dev_pos = []
train_dev_neg = []
test_pos = []
test_neg = []
for file in os.listdir("/Users/rizwanparvez/Downloads/aclImdb/train/pos"):
    if file.endswith(".txt"):
        # print(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/pos", file))
        id,score = file.split("_")
        score,_ = score.split('.')
        # print(int(id), int(score))
        with open(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/pos", file), 'r') as f:
            train_dev_pos.append("0."+score + "\t" + f.readline().replace("\n", ""))

for file in os.listdir("/Users/rizwanparvez/Downloads/aclImdb/train/neg"):
    if file.endswith(".txt"):
        # print(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/pos", file))
        id,score = file.split("_")
        score,_ = score.split('.')
        # print(int(id), int(score))
        with open(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/neg", file), 'r') as f:
            train_dev_neg.append("0."+score + "\t" + f.readline().replace("\n", ""))

for file in os.listdir("/Users/rizwanparvez/Downloads/aclImdb/test/pos"):
    if file.endswith(".txt"):
        # print(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/pos", file))
        id,score = file.split("_")
        score,_ = score.split('.')
        # print(int(id), int(score))
        with open(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/test/pos", file), 'r') as f:
            test_pos.append("0."+score + "\t" + f.readline().replace("\n", ""))

for file in os.listdir("/Users/rizwanparvez/Downloads/aclImdb/test/neg"):
    if file.endswith(".txt"):
        # print(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/train/pos", file))
        id,score = file.split("_")
        score,_ = score.split('.')
        # print(int(id), int(score))
        with open(os.path.join("/Users/rizwanparvez/Downloads/aclImdb/test/neg", file), 'r') as f:
            test_neg.append("0."+score+"\t"+ f.readline().replace("\n", ""))


print(len(train_dev_pos))
print(len(train_dev_neg))
print(len(test_pos))
print(len(test_neg))

dev = []
dev = np.random.choice(len(train_dev_pos), 1500)
trp = [train_dev_pos[i] for i in range(len(train_dev_pos)) if i not in dev]
trn = [train_dev_neg[i] for i in range(len(train_dev_neg)) if i not in dev]
devp = [train_dev_pos[i] for i in dev]
devn = [train_dev_neg[i] for i in dev]

with open('/Users/rizwanparvez/Downloads/aclImdb/train_imdb.txt', 'w') as trf:
    for i in range(len(trp)):
        line = trp[i]
        trf.write(line+'\n')
        line = trn[i]
        trf.write(line+'\n')
with open('/Users/rizwanparvez/Downloads/aclImdb/dev_imdb.txt', 'w') as trf:
    for i in range(len(devp)):
        line = devp[i]
        trf.write(line+'\n')
        line = devn[i]
        trf.write(line+'\n')
with open('/Users/rizwanparvez/Downloads/aclImdb/test_imdb.txt', 'w') as trf:
    for i in range(len(devp)):
        line = test_pos[i]
        trf.write(line+'\n')
        line = test_neg[i]
        trf.write(line+'\n')
