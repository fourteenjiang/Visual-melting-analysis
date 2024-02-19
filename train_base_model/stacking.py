#!/users/fourteen/.conda/envs/py39/bin/python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import timeit
import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
# from dataloaders.dataset import VideoDataset
from dataset import VideoDataset
from pytorchtools import EarlyStopping
import time
import numpy as np
import matplotlib.pyplot as plt
from mypath import Path
from sklearn.model_selection import KFold, StratifiedKFold
import shutil
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device being used:", device)
batch_size = 16
trial = 1

Log_result = ('fusion.txt')


def get_label_and_video_name(path):
    """read label txt file to extract video name and label
    """
    label = []
    video_name = []
    data = open(path, 'r')
    data_lines = data.readlines()

    for data_line in data_lines:
        data = data_line.strip().split(' ')
        video_name.append(data[0])
        if data[1] == 'solid':
            label.append(0)
        elif data[1] == 'half':
            label.append(1)
        else:
            label.append(2)
    return label, video_name


def create_meta_features(model, dataloader):
    model.eval()
    meta_features = []

    for inputs, _ in dataloader:
        inputs = Variable(inputs).to(device)
        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        meta_features.extend(probs.cpu().numpy())

    return np.array(meta_features)


def write_log(meg):
    with open(Log_result, 'a') as f:
        f.write(meg + '\n')


def load_model(model, trial):
    # Load a specific model given its name.
    path = os.path.join(model.split('_')[0] + "_result " + model.split('_')[1], "FT",
                        model.split('_')[0] + "_"+str(trial) +" trial.pth")
    model = torch.load(path)
    return model


# features =[R2Plus1D_basic_rgb_features, R2Plus1D_basic_gray_features, R3D_basic_rgb_features.....]

meta_features_val_all, meta_features_test_all = [], []
labels_train_each, labels_val_each,labels_test_each= [],[],[]
meta_features_test_all_each, meta_features_val_all_each,meta_features_train_all_each=[],[],[]

for modelName in ['R3D']:
    for type_video in ['gray', 'rgb']:
        # if modelName!='C3D' or type_video!='gray':

        fnames, labels = get_label_and_video_name(Path.db_dir('rgb')[2])[1], \
                         get_label_and_video_name(Path.db_dir('rgb')[2])[0]
        fnames, fnames_meta, labels, labels_meta = train_test_split(fnames, labels, test_size=0.2,
                                                                              random_state=41, stratify=labels)
        train_dataloader = DataLoader(
            VideoDataset(split='train', fnames=fnames_meta, labels=labels_meta, type_video=type_video, clip_len=16,
            process=True), batch_size=batch_size, num_workers=0)
        

        model = modelName + "_" + type_video
        model_instance = load_model(model,trial)
        meta_features_train = create_meta_features(model_instance, train_dataloader)
        meta_features_train_all_each.append(meta_features_train)
        if type_video =='rgb'  :

            labels_train_each += labels_meta


        
        skf = StratifiedKFold(n_splits=5, random_state=41, shuffle=True)

        for train_index, val_index in skf.split(X=fnames, y=labels):
            
            fnames_train, labels_train = [fnames[i] for i in train_index], [labels[i] for i in train_index]
            fnames_val, labels_val = [fnames[i] for i in val_index], [labels[i] for i in val_index]

            fnames_val, fnames_test, labels_val, labels_test = train_test_split(fnames_val, labels_val, test_size=0.5,
                                                                              random_state=41, stratify=labels_val)         
            
            
            val_dataloader = DataLoader(
                VideoDataset(split='val', fnames=fnames_val, labels=labels_val, type_video=type_video, clip_len=16,
                             process=True), batch_size=batch_size, num_workers=0)
            test_meta_dataloader = DataLoader(
                VideoDataset(split='test', fnames=fnames_test, labels=labels_test, type_video=type_video,
                             clip_len=16,
                             process=True),
                batch_size=batch_size, num_workers=0)
            # Load and train the model
            model = modelName + "_" + type_video
            model_instance = load_model(model,trial)


            meta_features_test = create_meta_features(model_instance, test_meta_dataloader)
            meta_features_val = create_meta_features(model_instance,val_dataloader)

            meta_features_test_all_each.append(meta_features_test)
            meta_features_val_all_each.append(meta_features_val)

            if type_video =='rgb'  :

                labels_val_each += labels_val
                labels_test_each += labels_test

            # if trial == 5:
            #     # meta_features_test = create_meta_features(model_instance, test_meta_dataloader)
            #     meta_features_test_5fold_average = np.mean(meta_features_test_all_each, axis=0)
            #     meta_features_test_all_each=[]
            #     meta_features_test_all.append(meta_features_test_5fold_average)
            #     trial = 0
            trial += 1

            shutil.rmtree(os.path.join(Path.db_dir(type_video)[1]))
        trial=1

print(len(labels_train_each),len(labels_test_each), len(labels_val_each))



X_train_meta = np.concatenate(meta_features_train_all_each, axis=0)
print('X_train_meta.reshape', np.array(X_train_meta).shape, type(np.array(X_train_meta).shape))

X_train_meta = np.stack([X_train_meta[i:i + len(labels_train_each)] for i in range(0, len(labels_train_each) * 2, len(labels_train_each))], axis=1)
# X_test_meta = np.stack((X_test_meta[:120], X_test_meta[120:240], X_test_meta[240:360]), axis=1)
X_train_meta = X_train_meta.reshape(X_train_meta.shape[0], -1)
print('X_train_meta.reshape', np.array(X_train_meta).shape)


meta_model_R = RandomForestClassifier(n_estimators=30, min_samples_split=2,
                                      random_state=42)  # n_estimators=100 0.9752 150 0.9667
meta_model_D = DecisionTreeClassifier(min_samples_split=2, random_state=42)
meta_model_S = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
meta_model_MLP = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42)
meta_model_L = LogisticRegression(C=1.0, penalty='l2', random_state=40)

for meta_model in [ meta_model_L]:
    param_grid = {
    'C': np.logspace(-2, 0, 40),  
    'penalty': ['l1', 'l2']  
}


    log_reg = LogisticRegression(random_state=40, solver='liblinear', max_iter=100)  # max_iter

  
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search.fit(X_train_meta, labels_train_each)
    
    best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validated accuracy: ", grid_search.best_score_)
    dump(best_model, str(meta_model) + '.joblib')
    model_file_size = os.path.getsize( str(meta_model) + '.joblib')
    print(f"Model file size: {model_file_size} bytes")





No_test=int(len(labels_test_each)/5)
X_test_meta = np.concatenate(meta_features_test_all_each, axis=0)
print('X_test_meta.reshape', np.array(X_test_meta).shape, type(np.array(X_test_meta).shape))



f1_scores = []  

for i in range(5):
    X_test_meta_= np.concatenate([X_test_meta[ i*No_test:(i+1)*No_test],X_test_meta[ ((5+i)*No_test):((i+6)*No_test)]],axis=0)
    X_test_meta_ = np.stack([X_test_meta_[i:i + No_test] for i in range(0, No_test * 2, No_test)], axis=1) 
    X_test_meta_ = X_test_meta_.reshape(X_test_meta_.shape[0], -1)
    print('X_test_meta.reshape', np.array(X_test_meta_).shape)


    predictions = best_model.predict(X_test_meta_)
    label=labels_test_each[ i*No_test:(i+1)*No_test]+labels_test_each[ ((5+i)*No_test):((i+6)*No_test)]
    accuracy = accuracy_score(label,
                              predictions)  # Assuming labels_test_meta_combined contains all the test labels
    f1 = f1_score(label, predictions, average='weighted')
    f1_scores.append(f1)
    write_log(str(meta_model))
    print(f"Stacking Ensemble Accuracy: {accuracy:.4f}")
    write_log(f"Stacking Ensemble Accuracy: {accuracy:.4f}")
    print(f"Stacking Ensemble f1 score: {f1:.4f}")
    write_log(f"Stacking Ensemble f1 score: {f1:.4f}")

f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

print(f"Mean F1 Score: {f1_mean:.4f}")
write_log(f"Mean F1 Score: {f1_mean:.4f}")
print(f"F1 Score Standard Deviation: {f1_std:.4f}")
write_log(f"F1 Score Standard Deviation: {f1_std:.4f}")



