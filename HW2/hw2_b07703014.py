'''
Financial Technology
Assignment 2
B07703014 蔡承翰
Due 16-11-2020
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import ensemble
from sklearn import tree

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Train on GPU or CPU

# DNN model
class Network(nn.Module):
    def __init__(self, hidden_layer: int, unit: int):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.unit = unit
        self.input_node = 30
        self.output_node = 1  # Binary classification
        self.model = nn.Sequential()
        self.get_model()
    
    def get_model(self):
        n_neuron = self.input_node
        for layer in range(self.hidden_layer - 1):
            self.model.add_module(("fc_" + str(layer + 1)), nn.Linear(n_neuron, self.unit))
            self.model.add_module(("Relu_" + str(layer + 1)), nn.ReLU(inplace = True))
            n_neuron = self.unit
        self.model.add_module(("fc_" + str(self.hidden_layer)), nn.Linear(n_neuron, self.output_node))
        self.model.add_module("Sig", nn.Sigmoid())  # sigmoid tranformation at the final output
    
    def forward(self, x):
        return self.model(x)

'''------------------------------------------Functions-------------------------------------------------------'''

def bin_accuracy(y_hat: torch.tensor, y: torch.tensor):
    '''Compute the accuracy given the prediction and the actual value'''
    y = y.cpu()
    prediction = torch.round(y_hat.cpu())
    return precision_score(y.detach().numpy(), prediction.detach().numpy())

def get_processed_data():
    '''Return preprocessed data in ndarray'''
    import random as rand
    rand.seed(20201116)
    raw_data = pd.read_csv("Data.csv")
    row_count = len(raw_data.index)
    testset_idxes = rand.sample(list(raw_data.index), round(row_count * 0.2))
    test_df = raw_data.iloc[testset_idxes]
    train_df = raw_data.drop(testset_idxes)

    train_y_arr = np.expand_dims(train_df["Class"].to_numpy(), axis = 1)
    train_x_arr = train_df.drop(columns = ["Class"]).to_numpy()
    test_y_arr = np.expand_dims(test_df["Class"].to_numpy(), axis = 1)
    test_x_arr = test_df.drop(columns = ["Class"]).to_numpy()

    test_x_arr = (test_x_arr - np.mean(train_x_arr, axis = 0)) / np.std(train_x_arr, axis = 0)
    train_x_arr = (train_x_arr - np.mean(train_x_arr, axis = 0)) / np.std(train_x_arr, axis = 0)

    return train_x_arr, train_y_arr, test_x_arr, test_y_arr

def train_DNN(epochs: int, lr: float, trainloader: DataLoader, testloader: DataLoader, hidden_layer: int, unit: int):
    '''Train DNN model with given parameters'''
    net = Network(hidden_layer, unit)
    net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr = lr)  # An Adam optimizer
    criterion = nn.BCELoss()
    epoch_idx = []
    train_epoch_loss = []
    test_epoch_loss = []
    train_epoch_accuracy = []
    test_epoch_accuracy = []
    # Training
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for b_num, (x, y) in enumerate(trainloader):
            x = x.to(device)  # get x, y onto GPU
            y = y.to(device)
            optimizer.zero_grad()  # Clear the gradient
            y_hat = net(x) # Forward
            loss = criterion(y_hat, y)
            acc = bin_accuracy(y_hat, y)

            # Backward prop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()
        
        # Trainset evaluation
        # loader = DataLoader(train_dataset, batch_size = size, shuffle = False, pin_memory = True)
        train_y, train_pred = get_all_preds(net, trainloader)
        train_acc = precision_score(train_y, train_pred)
        train_epoch_accuracy.append(train_acc)
        epoch_idx.append(epoch + 1)  # epoch: 0 ~ N-1
        train_epoch_loss.append(train_loss)
        print("epoch: " + str(epoch + 1) + "\tLoss: " + str(round(train_loss, 2)) + "\tAccurancy: " + str(round(train_acc, 2)), end = "\t")
        
        # Testset evaluation
        test_loss = 0
        test_acc = 0
        for b_num, (x, y) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            acc = bin_accuracy(y_hat, y)
            test_loss += loss.item()
            test_acc += acc.item()
        test_y, test_pred = get_all_preds(net, testloader)
        test_acc = precision_score(test_y, test_pred)
        test_epoch_accuracy.append(test_acc)
        test_epoch_loss.append(test_loss)
        print("\tValidation Loss: " + str(round(test_loss, 2)) + "\t\tValidation Accurancy: " + str(round(test_acc, 2)))

    return net, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy
            
def get_all_preds(net: Network, dataloader: DataLoader, is_round = True):
    all_preds = torch.Tensor([])
    all_y = torch.Tensor([])
    for b_num, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y_hat = net(x).cpu()
        all_preds = torch.cat((all_preds, y_hat), dim = 0)
        all_y = torch.cat((all_y, y), dim = 0)
    if is_round:
        all_preds = torch.round(all_preds)
    return all_y.detach().numpy(), all_preds.detach().numpy()

def draw_confusion_matrix(cm, title):
    names = ["0", "1"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap = "Blues")
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + names)
    ax.set_yticklabels([''] + names)
    plt.text(0.3, 0.7, str(cm[0, 0]), horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
    plt.text(0.7, 0.7, str(cm[0, 1]), horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
    plt.text(0.3, 0.3, str(cm[1, 0]), horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
    plt.text(0.7, 0.3, str(cm[1, 1]), horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
    plt.xlabel("Prediction")
    plt.ylabel("True value")
    plt.show()

def get_precision_recall_F1(cm, class_name):
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    F1 = 2 * ((precision * recall) / (precision + recall))
    print()
    print("Class: " + class_name)
    print("Precision: " + str(round(precision, 3)) + "\tRecall: " + str(round(recall, 3)) + "\tF1-score: " + str(round(F1, 3)))
    return precision, recall, F1

def get_roc_prc(y_arr, y_hat_arr):
    fprs, tprs, thres = roc_curve(y_arr, y_hat_arr)
    auc = roc_auc_score(y_arr, y_hat_arr)
    precision, recall, thres = precision_recall_curve(y_arr, y_hat_arr)
    ave_pre_score = average_precision_score(y_arr, y_hat_arr)
    return fprs, tprs, auc, precision, recall, ave_pre_score

def get_lift_curve(y_arr, y_hat_arr, y_score_arr, DNN = False):
    y_list = np.transpose(y_arr).tolist()[0]
    y_hat_list = np.transpose(y_hat_arr).tolist()
    y_score_list = np.transpose(y_score_arr).tolist()
    if DNN:
        y_hat_list = y_hat_list[0]
        y_score_list = y_score_list[0]
    y_list = [y for _,y in sorted(zip(y_score_list, y_list), reverse = True)]
    y_hat_list = [y_hat for _,y_hat in sorted(zip(y_score_list, y_hat_list), reverse = True)]
    accum_TPs = []
    accum_TP = 0
    for i in range(len(y_list)):
        if y_list[i] == y_hat_list[i] and int(y_list[i]) == 1:
            accum_TP += 1
        accum_TPs.append(accum_TP)
    return accum_TPs

'''------------------------------------------Main--------------------------------------------------------'''
batch_sizes = [128, 256, 512]  # All tried value
epochs = [1, 25, 50]
lrs = [0.01, 0.001]
hidden_layers = [3, 4, 5]
units = [16, 32, 64]


train_x_arr, train_y_arr, test_x_arr, test_y_arr = get_processed_data()
train_dataset = TensorDataset(torch.Tensor(train_x_arr), torch.Tensor(train_y_arr))
test_dataset = TensorDataset(torch.Tensor(test_x_arr), torch.Tensor(test_y_arr))

# size = 512
# epochs = 50
trainloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True)
testloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, pin_memory = True)

# Grid search to train parameters
best_size = int()
best_epoch = int()
best_lr = float()
best_acc = 0.0
best_hidden_layer = int()
best_unit = int()
best_train_losses = []
best_train_accs = []
best_test_losses = []
best_test_accs = []
PATH = "best_model_para.pt"  # Path to save model parameters

for size in batch_sizes:
    trainloader = DataLoader(train_dataset, batch_size = size, shuffle = True, pin_memory = True)
    testloader = DataLoader(test_dataset, batch_size = size, shuffle = True, pin_memory = True)
    for epoch in epochs:
        for lr in lrs:
            for hidden_layer in hidden_layers:
                for unit in units:
                    print()
                    print("Size: " + str(size) + "\tLearning Rate: " + str(lr) + "\tn_epoch: " + str(epoch) + "\tHidden layers: " + str(hidden_layer) + "\tUnit: " + str(unit))
                    net, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy = train_DNN(epoch, lr, trainloader, testloader, hidden_layer, unit)
                    if test_epoch_accuracy[-1] > best_acc:  # Maximize validation accurancy
                        best_size = size
                        best_epoch = epoch
                        best_lr = lr
                        best_hidden_layer = hidden_layer
                        best_unit = unit
                        best_acc = test_epoch_accuracy[-1]
                        best_train_losses = train_epoch_loss
                        best_train_accs = train_epoch_accuracy
                        best_test_losses = test_epoch_loss
                        best_test_accs = test_epoch_accuracy
                        torch.save(net.state_dict(), PATH)

print()
print("Optimal size: " + str(best_size) + "\t Optimal lr: " + str(best_lr) + "\tOptimal iteration times: " + str(best_epoch))
print("Optimal hidden layer: " + str(best_hidden_layer) + "\tOpitmal units of neuron: " + str(best_unit))
print("(a) Achieved training accurancy: " + str(round(best_train_accs[-1], 3)))
print("(b) Achieved testing accuancy: " + str(round(best_acc, 3)))
print("(c) Achieved training loss: " + str(round(best_train_losses[-1], 3)))
print("(d) Achieved testing loss: " + str(round(best_test_losses[-1], 3)))

# Loss diagram
epoch_idx = range(best_epoch)
plt.plot(epoch_idx, best_train_losses, label = "Train")
plt.plot(epoch_idx, best_test_losses, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# Accuracy diagram
plt.plot(epoch_idx, best_train_accs, label = "Train", linewidth = 2)
plt.plot(epoch_idx, best_test_accs, label = "Validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()


# Reload best model
best_state_dict = torch.load(PATH)
model = Network(best_hidden_layer, best_unit)
model.load_state_dict(best_state_dict)
model.to(device)
print(model)

with torch.no_grad():
    pred_trainloader = DataLoader(train_dataset, batch_size = best_size, shuffle = False, pin_memory = True)
    pred_testloader = DataLoader(test_dataset, batch_size = best_size, shuffle = False, pin_memory = True)

# Confusion matrix of training set
train_y, train_pred = get_all_preds(model, pred_trainloader)
train_cm = confusion_matrix(train_y, train_pred)
print(train_cm)
draw_confusion_matrix(train_cm, "Confusion matrix of training set")

# Confusion matrix of testing set
test_y, test_pred = get_all_preds(model, pred_testloader)
test_cm = confusion_matrix(test_y, test_pred)
print(test_cm)
draw_confusion_matrix(test_cm, "Confusion matrix of validation set")

# Precicion and Recall
# Class: Train
train_precision, train_recall, train_F1 = get_precision_recall_F1(train_cm, "Train")
# Class: Validation
test_precision, test_recall, test_F1 = get_precision_recall_F1(test_cm, "Validation")
# Class: Average
ave_precision, ave_recall, ave_F1 = get_precision_recall_F1((train_cm + test_cm), "Average")


# Random Forest
forest = ensemble.RandomForestClassifier()
forest.fit(train_x_arr, train_y_arr)
forest_test_pred = forest.predict_proba(test_x_arr)[:, 1]
forest_fprs, forest_tprs, forest_auc, forest_precision, forest_recall, forest_AP = get_roc_prc(test_y_arr, forest_test_pred)
forest_test_acc = accuracy_score(test_y_arr, forest.predict(test_x_arr))
forest_precision_score = precision_score(test_y_arr, forest.predict(test_x_arr))
forest_recall_score = recall_score(test_y_arr, forest.predict(test_x_arr))
forest_f1 = f1_score(test_y_arr, forest.predict(test_x_arr))
print("Random Forest")
print("Accuracy: " + str(round(forest_test_acc, 3)) + "\tPrecision: " + str(round(forest_precision_score, 3)) + "\t Recall: " + str(round(forest_recall_score, 3)) + "\tF1-score: " + str(round(forest_f1, 3)))
print()

# Decision Tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(train_x_arr, train_y_arr)
tree_test_pred = decision_tree.predict_proba(test_x_arr)[:, 1]
tree_fprs, tree_tprs, tree_auc, tree_precision, tree_recall, tree_AP = get_roc_prc(test_y_arr, tree_test_pred)
tree_test_acc = accuracy_score(test_y_arr, decision_tree.predict(test_x_arr))
tree_precision_score = precision_score(test_y_arr, decision_tree.predict(test_x_arr))
tree_recall_score = recall_score(test_y_arr, decision_tree.predict(test_x_arr))
tree_f1 = f1_score(test_y_arr, decision_tree.predict(test_x_arr))
print("Decision Tree")
print("Accuracy: " + str(round(tree_test_acc, 3)) + "\tPrecision: " + str(round(tree_precision_score, 3)) + "\t Recall: " + str(round(tree_recall_score, 3)) + "\tF1-score: " + str(round(tree_f1, 3)))
print()


# DNN
train_y, train_y_hat_no_round = get_all_preds(model, pred_trainloader, is_round = False)
test_y, test_y_hat_no_round = get_all_preds(model, pred_testloader, is_round = False)
fprs, tprs, auc_score, precision, recall, ave_pre_score = get_roc_prc(test_y, test_y_hat_no_round)

# ROC and PRC
plt.plot(fprs, tprs, label = ("DNN AUC = " + str(round(auc_score, 4))), linewidth = 2)
plt.plot(tree_fprs, tree_tprs, label = ("Decision Tree AUC = " + str(round(tree_auc, 4))))
plt.plot(forest_fprs, forest_tprs, label = ("Random Forest AUC = " + str(round(forest_auc, 4))))
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()

plt.plot(recall, precision, label = "DNN AP = " + str(round(ave_pre_score, 4)))
plt.plot(tree_recall, tree_precision, label = "Decision Tree AP = " + str(round(tree_AP, 4)))
plt.plot(forest_recall, forest_precision, label = "Random Forest AP = " + str(round(forest_AP, 4)))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Lift Curve
train_x, train_y_hat = get_all_preds(model, pred_trainloader)
test_x, test_y_hat = get_all_preds(model, pred_testloader)
# DNN_y_arr = np.concatenate((train_x, test_x), axis = 0)
# DNN_y_hat_arr = np.concatenate((train_y_hat, test_y_hat),axis = 0)
DNN_accum_TPs = get_lift_curve(test_y_arr, test_y_hat, test_y_hat_no_round, DNN = True)

# x_arr = np.concatenate((train_x_arr, test_x_arr), axis = 0)
# y_arr = np.concatenate((train_y_arr, test_y_arr), axis = 0)
tree_accum_TPs = get_lift_curve(test_y_arr, decision_tree.predict(test_x_arr), decision_tree.predict_proba(test_x_arr)[:, 1])

forest_accum_TPs = get_lift_curve(test_y_arr, forest.predict(test_x_arr), forest.predict_proba(test_x_arr)[:, 1])

ideal_TPs = get_lift_curve(test_y_arr, test_y_arr, test_y_arr, DNN = True)

random_bin_arr = np.random.rand(len(DNN_accum_TPs), 1)
random_TPs = get_lift_curve(test_y_arr, np.round(random_bin_arr), random_bin_arr, DNN = True)


samples = range(1, len(DNN_accum_TPs) + 1)

plt.plot(samples, DNN_accum_TPs, label = "DNN", linewidth = 4)
plt.plot(samples, tree_accum_TPs, label = "Decision Tree", linewidth = 3)
plt.plot(samples, forest_accum_TPs, label = "Random Forest", linewidth = 2)
plt.plot(samples, ideal_TPs, label = "Ideal model")
plt.plot(samples, random_TPs, label = "Random Guess")
plt.xlabel("Sample")
plt.ylabel("Accumulative True Positive")
plt.title("Lift Curve")
plt.legend()
plt.show()


print("Done")






