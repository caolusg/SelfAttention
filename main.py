import dataProcessing
import hyperparameters
import train
import modelElite
import modelElite_LSTM
import modelElite_cnn
import modelElite_BiLSTM
import torch


classifier = dataProcessing.Classifier()

train_iter, dev_iter, test_iter, InitRst_test = classifier.firstDataProcessing(data="./data/subj.all",
                                                                               hy=classifier.hyperparameter)

model = modelElite_BiLSTM.model(classifier.hyperparameter)
train.train(train_iter, dev_iter, test_iter, model, classifier.hyperparameter)

if classifier.hyperparameter.snapshot is not None:
    print("Loading model from [%s]..." % classifier.hyperparameter.snapshot)
    model = torch.load(classifier.hyperparameter.snapshot)
    train.test_eval(test_iter, model, classifier.hyperparameter, InitRst_test)