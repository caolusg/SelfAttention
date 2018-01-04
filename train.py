from torch.autograd import Variable
import dataProcessing
import random
import torch
import os
import torch.nn.functional as F
from testMatplotlib import vision
from testdrawer import drawer
import modelElite
import modelElite_cnn
import modelElite_LSTM
import hyperparameters
from colortest import testPd


def eval(data_iter, model, hyperparameter, test):
    runTimeList = []
    # model.eval()
    total_num = len(data_iter)
    print("dev num:", total_num)

    part = total_num // hyperparameter.batch
    if total_num % hyperparameter.batch != 0:
        part += 1
    for n in range(1):

        # print('第%d次迭代：' % (n + 1))
        correct = 0
        sum = 0
        for idx in range(part):
            begin = idx * hyperparameter.batch
            end = (idx + 1) * hyperparameter.batch
            if end > total_num:
                end = total_num
            batch_list = []
            for idy in range(begin, end):
                batch_list.append(data_iter[idy])

            random.shuffle(batch_list)
            # print('第%d次迭代：' % (n + 1))
            #correct = 0
            #sum = 0
            # print("fe", batch_list[0])
            # optimizer.zero_grad()

            feature, target = ToVariable(batch_list, hyperparameter)
            if hyperparameter.LSTM_model:
                if feature.size(0) == hyperparameter.batch :
                    model.hidden = model.init_hidden(hyperparameter.num_layers, hyperparameter.batch)

                else :
                    model.hidden = model.init_hidden(hyperparameter.num_layers, feature.size(0))

            # print("fea",feature)
            logit, alpha = model(feature)
            # print(logit)
            loss = F.cross_entropy(logit, target)  # 目标函数求导
            print("idx", idx, "dev loss:", loss.data[0])
            # loss.backward()
            # optimizer.step()
            for i in range(len(target)):
                # print(logit[i])
                # print(logit)
                if target.data[i] == getMaxIndex(logit[i].view(1, hyperparameter.labelSize)):
                    correct += 1
                    # print(""correct)
                # print('loss:',loss.data[0])
                # print(correct)

                sum += 1
            #print(sum)
        if test :
            print("test", end=" ")
        else :
            print("dev", end=" ")
        print("correct number:", correct)
        acc = correct / sum
        print('acc:', acc)

    return acc


def test_eval(data_iter, model, hyperparameter, InitRst_test):
    total_num = len(data_iter)
    print("eval_num:", total_num)
    hyperparameter.batch = 1
    batch = hyperparameter.batch

    if total_num % batch == 0:
        num_batch = total_num // batch
    else:
        num_batch = total_num // batch + 1

    # for x in range(1, 2):
    # random.shuffle(data_iter)
    # print('这是第%d次迭代' % x)
    correct = 0
    sum = 0
    avg_loss = 0
    steps = 0
    for i in range(num_batch):
        batch_list = []
        allwords=[]
        for j in range(i * batch,
                       (i + 1) * batch if (i + 1) * batch < len(data_iter) else len(
                           data_iter)):
            batch_list.append(data_iter[j])

        random.shuffle(batch_list)  # 进行重新洗牌
        feature, target = ToVariable(batch_list, hyperparameter)

        if hyperparameter.LSTM_model:
            if feature.size(0) == hyperparameter.batch:
                model.hidden = model.init_hidden(hyperparameter.num_layers,
                                                 hyperparameter.batch)
            else:
                model.hidden = model.init_hidden(hyperparameter.num_layers,
                                                 feature.size(0))
        # optimizer.zero_grad()
        logit, alpha = model(feature)
        # print(alpha.data.tolist()[0])
        # print(InitRst_test[i].m_word[0])

        drawer(alpha.data.tolist()[0], InitRst_test[i].m_word[0])



        loss = F.cross_entropy(logit, target, size_average=False)  # 目标函数的求导
        avg_loss += loss.data[0]
        # print('loss:',loss.  data[0])
        for i in range(len(target)):
            if (target.data[i] == getMaxIndex(logit[i].view(1, hyperparameter.labelSize))):
                correct += 1
            sum += 1
    size = len(data_iter)
    avg_loss = loss.data[0] / size
    accuracy = correct / sum
    # print('eval acc:{} correct / sum {} / {}'.format(correct / sum, correct, sum))

    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       correct,
                                                                       sum))


def train(Example_list_train, Example_list_dev, Example_list_test, model, hyperparameter):

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter.lr)
    total_num = len(Example_list_train)
    print("train num:", total_num)

    part = total_num // hyperparameter.batch
    if total_num % hyperparameter.batch != 0:
        part += 1

    model.train()
    steps = 0
    model_count = 0
    listOfTestAcc = []
    listOfDevAcc = []
    for n in range(hyperparameter.epoches):
        print('第%d次迭代：' % (n + 1))
        correct = 0
        sum = 0
        for idx in range(part):
            begin = idx * hyperparameter.batch
            end = (idx + 1) * hyperparameter.batch
            if end > total_num:
                end = total_num
            batch_list = []
            for idy in range(begin, end):
                batch_list.append(Example_list_train[idy])

            random.shuffle(batch_list)

            optimizer.zero_grad()
            model.zero_grad()
            feature, target = ToVariable(batch_list, hyperparameter)
            if hyperparameter.LSTM_model:
                if feature.size(0) == hyperparameter.batch :
                    model.hidden = model.init_hidden(hyperparameter.num_layers, hyperparameter.batch)

                else :
                    model.hidden = model.init_hidden(hyperparameter.num_layers, feature.size(0))

            # print("fea",feature)
            logit, alpha = model(feature)

            loss = F.cross_entropy(logit, target)  # 目标函数求导
            # print(loss)
            print("idx", idx, "loss:", loss.data[0])
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            steps += 1
            for i in range(len(target)):
                # print(logit[i])
                # print(logit)
                if target.data[i] == getMaxIndex(logit[i].view(1, hyperparameter.labelSize)):
                    correct += 1
                    # print(""correct)
                # print('loss:',loss.data[0])
                # print(correct)

                sum += 1
            #print(sum)
        print("train", end=" ")
        print("correct number:", correct)
        print('acc:', correct / sum)

        devAcc = eval(Example_list_dev, model, hyperparameter, test=False)
        listOfDevAcc.append(devAcc)

        if not os.path.isdir(hyperparameter.save_dir): os.makedirs(hyperparameter.save_dir)
        save_prefix = os.path.join(hyperparameter.save_dir, 'snapshot')
        save_path = '{}_steps{}.pt'.format(save_prefix, n)
        torch.save(model, save_path)
        print("\n", save_path, end=" ")
        test_model = torch.load(save_path)
        model_count += 1

        # def test_eval(data_iter, model, save_path, args, model_count):

        testAcc = eval(Example_list_test, test_model, hyperparameter, test=True)
        listOfTestAcc.append(testAcc)

    max = listOfTestAcc[0]
    maxDev = listOfDevAcc[0]
    # location = 0
    for i in listOfTestAcc:
        if i > max :
            max = i
    for n in range(len(listOfTestAcc)):
        if listOfTestAcc[n] == max :
            print("max acc located in ", n+1)
            # break

    for j in listOfDevAcc:
        if j > maxDev :
            maxDev = j
    for m in range(len(listOfTestAcc)):
        if listOfTestAcc[m] == maxDev :
            print("max acc located in ", m+1)
            # break

    print("test max acc: ", max)
    print("dev  max acc: ", maxDev)


def ToVariable(Examples, hyperparameter):  # 输入为包含batch个example的list

    batch = len(Examples)
    maxLength = 0
    for i in range(len(Examples)):
        if len(Examples[i].word_indexes) > maxLength:
            maxLength = len(Examples[i].word_indexes)

    x = Variable(torch.LongTensor(batch, maxLength))
    y = Variable(torch.LongTensor(batch))
    for i in range(len(Examples)):
        for n in range(len(Examples[i].word_indexes)):
            x.data[i][n] = Examples[i].word_indexes[n]
            for j in range(len(Examples[i].word_indexes), maxLength):
                x.data[i][j] = hyperparameter.unknown
        y.data[i] = Examples[i].label_index[0]
        # print("Y:", y)

    return x, y

def getMaxIndex(score):
    # print("sss", score)
    labelsize = score.size()[1]
    max = score.data[0][0]
    maxIndex = 0
    for idx in range(labelsize):
        tmp = score.data[0][idx]
        if max < tmp:
            max = tmp
            maxIndex = idx

    return maxIndex
