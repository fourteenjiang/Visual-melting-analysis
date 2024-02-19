#!/users/fourteen/.conda/envs/py39/bin/python
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
import C3D_model, R2Plus1D_model, R3D_model, R3D_fusion, R2Plus1D_fusion, C3D_fusion
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

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device being used:", device)

nEpochs = 100 # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 20  # Run on test set every nTestInterval epochs
snapshot = 5  # Store a model every snapshot epochs
lr = 5.5e-2  # try Learning rate 1. p=18 lr=5e-2
num_classes = 3
patience = 18 #1.p=18 
delta = 5e-3
batch_size = 8
trial = 0
type_video = 'gray'
modelName = 'C3D'  # Options: C3D or R2Plus1D or R3D
save_dir = os.path.join(os.getcwd(), modelName + '_result ' + type_video)
early=True
FE=False


def train_model(model, optimizer, scheduler, trainval_loaders,  extractor, test_loaders,num_epochs=nEpochs, patience=patience, delta=delta, early=early):
    """
    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        trainval_loaders (dict): Dictionary containing DataLoader for training and validation data.
        num_epochs (int, optional): Number of epochs to train for.
        patience (int, optional): Patience for early stopping.
        delta (float, optional): Delta for early stopping.
        resume_epoch (int, optional): The starting epoch when resuming training.

    Returns:
        tuple: A tuple containing the trained model, validation accuracy history, training history, and best accuracy.
    """

    since = time.time()
    val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {
        'train': {'loss': [], 'acc': []},
        'val': {'loss': [], 'acc': []}
    }
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)
    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs_ir,inputs_rgb, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs_ir = Variable(inputs_ir, requires_grad=True).to(device)
                inputs_rgb = Variable(inputs_rgb, requires_grad=True).to(device)
                labels = Variable(labels).to(device).long()
                optimizer.zero_grad()
                early_feature, late_feature = extractor.extract_features(inputs_ir)
                for param in extractor.parameters():
                    param.requires_grad = False

                if phase == 'train':

                    #early_feature = early_feature.to(device)
                    #late_feature = late_feature.to(device)
                    if early:
                       logits, early_output, late_output = model(inputs_rgb, early_feature, late_fusion_feature=None)
                    else:
                        logits, early_output, late_output = model(inputs_rgb, early_fusion_feature=None, late_fusion_feature=late_feature)
                    outputs = logits
                else:
                    with torch.no_grad():

                        if early:
                            logits, early_output, late_output = model(inputs_rgb, early_feature, late_fusion_feature=None)
                        else:
                            logits, early_output, late_output = model(inputs_rgb, early_fusion_feature=None, late_fusion_feature=late_feature)
                        outputs = logits

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                running_loss += loss.item() * inputs_rgb.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)
            if phase == 'val':
                valid_loss = epoch_loss
                val_acc_history.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint.pt'))
                
            test_model(model=model, test_dataloader=test_loaders,early=early, extractor=extractor, save=False)

            print("[{}] Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # write_log("Early stopping")
            break

    time_elapsed = time.time() - since
    write_log('Best val Acc: {:4f}'.format(best_acc))
    write_log('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # write_log('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint.pt')))
    torch.save(model, (os.path.join(save_dir, '{}_{} trial.pth'.format(modelName, trial))))
    write_log('Saving {}_{} trial.pth'.format(modelName, trial))
    write_log('Trial {} finished'.format(trial))
    return model, val_acc_history, history, best_acc


def test_model(model, test_dataloader, early, extractor, save ):
    test_corrects = 0
    test_total = 0
    test_predictions = []
    test_true_labels = []
    model.eval()  # Set the model to evaluation mode
    extractor.eval()
    for inputs_ir, inputs_rgb, labels in tqdm(test_dataloader):
        # move inputs and labels to the device the training is taking place on
        inputs_ir = Variable(inputs_ir, requires_grad=False).to(device)
        inputs_rgb = Variable(inputs_rgb, requires_grad=False).to(device)
        labels = Variable(labels).to(device).long()



        with torch.no_grad():
            early_feature, late_feature = extractor.extract_features(inputs_ir)

            if early:
                logits, early_output, late_output = model(inputs_rgb, early_feature, late_fusion_feature=None)
            else:
                logits, early_output, late_output = model(inputs_rgb,early_fusion_feature=None, late_fusion_feature=late_feature)
            outputs = logits

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]


        test_predictions.extend(preds.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

        test_corrects += torch.sum(preds == labels.data)
        test_total += labels.size(0)

    test_accuracy = test_corrects.double() / test_total
    test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)
    f1 = f1_score(test_true_labels, test_predictions, average='weighted')

    
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print(str(test_conf_matrix))
    if save:
      f1score.append(f1.item())
      acc_ls.append(test_accuracy.item())
      write_log("Test Accuracy: {:.4f}".format(test_accuracy))
      write_log("Confusion Matrix for Test Data:")
      write_log(str(test_conf_matrix))
      write_log("F1 Score: {:.4f}".format(f1))


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print("Directory created successfully:", directory_path)
        except OSError as e:
            print("Failed to create directory:", directory_path)
            print("Error:", str(e))
    else:
        print("Directory already exists:", directory_path)


def write_log(meg):
    with open(Log_result, 'a') as f:
        f.write(meg + '\n')


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


def plot_history(history, trial):
    """Plot historical training and validation accuracies and losses

    Args:
        history (dict): training and validation losses and accuracies history.
                        {'train': {'loss': [], 'acc': []},
                         'val': {'loss': [], 'acc': []}}

    Returns:
        None
    """
    fig, ax1 = plt.subplots()

    # Correctly number epochs starting from 1
    epochs = np.arange(1, len(history['train']['loss']) + 1)

    # Plot losses
    ax1.plot(epochs, history['train']['loss'], 'g-')
    ax1.plot(epochs, history['val']['loss'], 'b-')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Validation Loss'], bbox_to_anchor=(0.6, 0.2))

    # find position of lowest validation loss
    minposs = history['val']['loss'].index(min(history['val']['loss'])) + 1

    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    path_name = '{}_{} trial'.format(modelName, trial)
    plt.title(path_name)
    PATH = '{}.png'.format(path_name)
    PATH = os.path.join(save_dir, PATH)
    fig.savefig(PATH, bbox_inches='tight')


def initialize_model(modelName, num_classes, lr, pretrained, FE):
    """
       Args:
           modelName (str): Name of the model to initialize.
           num_classes (int): Number of classes for the model.
           lr (float): Learning rate for the model.
           feature_extractor (boolean): Feature_extractor or Fine-tunning

       Returns:
           tuple: A tuple containing the initialized model and training parameters.
       """
    if modelName == 'C3D':
        extractor= C3D_fusion.C3DFeatureExtractor(num_classes=num_classes, pretrained=pretrained)
        model = C3D_fusion.C3D(num_classes=num_classes, pretrained=pretrained)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
        if FE:
            model = nn.Sequential(*list(model.children())[:-3])
            for param in model.parameters():
                param.requires_grad = False
            fc8 = nn.Linear(4096, num_classes, bias=True)
            model.add_module("fc8", fc8)
            model.add_module("dropout", nn.Dropout(0.5, inplace=False))
            model.add_module("relu", nn.ReLU())
            train_params = [{'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]




    elif modelName == 'R2Plus1D':
        extractor = R2Plus1D_fusion.R2Plus1DFeatureExtractor(num_classes=num_classes, layer_sizes=(2, 2, 2, 2),
                                                   pretrained=pretrained)
        model = R2Plus1D_fusion.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2),
                                                  pretrained=pretrained)

        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
        if FE:
            model = nn.Sequential(*list(model.children())[:-1])
            for param in model.parameters():
                param.requires_grad = False
            linear = nn.Linear(512, num_classes, bias=True)
            model.add_module("linear", linear)
            train_params = [{'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]


    elif modelName == 'R3D':  # 18

        extractor = R3D_fusion.R3DFeatureExtractor(num_classes=num_classes, layer_sizes=(2, 2, 2, 2), pretrained=pretrained)
        model= R3D_fusion.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2 , 2, 2), pretrained=pretrained )
        train_params = model.parameters()
        if FE:
            model = nn.Sequential(*list(model.children())[:-1])
            for param in model.parameters():
                param.requires_grad = False
            linear = nn.Linear(512, num_classes, bias=True)
            model.add_module("linear", linear)
            train_params = [{'params': R3D_model.get_10x_lr_params(model), 'lr': lr * 10}]



    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    return model, train_params, extractor


if __name__ == "__main__":
    for modelName in ['R3D', "R2Plus1D"]:
        for pretrained in [True]:
          for early in [ False, True]:
            acc_ls = []
            f1score = []
            save_dir = os.path.join(os.getcwd(), modelName , ' early ' if early else ' late',
                                    ' FT' if pretrained else 'full')
            check_and_create_directory(save_dir)
            Log_result = (os.path.join(save_dir, 'log.txt'))
            fnames, labels = get_label_and_video_name(Path.db_dir()[2][0])[1], \
                             get_label_and_video_name(Path.db_dir()[2][0])[0]
            print(labels)
            fnames, fnames_test_meta, labels, labels_test_meta = train_test_split(fnames, labels, test_size=0.2,
                                                                                  random_state=41, stratify=labels)

            skf = StratifiedKFold(n_splits=5, random_state=41, shuffle=True)

            for train_index, val_index in skf.split(X=fnames, y=labels):
                trial += 1
                write_log('\n' + 'TRIAL' + str(trial))
                fnames_train, labels_train = [fnames[i] for i in train_index], [labels[i] for i in train_index]
                fnames_val, labels_val = [fnames[i] for i in val_index], [labels[i] for i in val_index]

                fnames_val, fnames_test, labels_val, labels_test = train_test_split(fnames_val, labels_val,
                                                                                    test_size=0.5,
                                                                                    random_state=41,
                                                                                    stratify=labels_val)

                train_dataloader = DataLoader(
                    VideoDataset(split='train', fnames=fnames_train, labels=labels_train,
                                 clip_len=16,
                                 process=True), batch_size=batch_size, shuffle=True, num_workers=4)
                val_dataloader = DataLoader(
                    VideoDataset(split='val', fnames=fnames_val, labels=labels_val,  clip_len=16,
                                 process=True), batch_size=batch_size, num_workers=4)
                test_dataloader = DataLoader(
                    VideoDataset(split='test', fnames=fnames_test_meta, labels=labels_test_meta,
                                 clip_len=16,
                                 process=True),
                    batch_size=batch_size, num_workers=4)
                # test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=15), batch_size=batch_size, num_workers=4)

                trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
                trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
                # test_size = len(test_dataloader.dataset)
                model, parameters, extractor = initialize_model(modelName, num_classes=num_classes, lr=lr, pretrained=pretrained,
                                                     FE=FE)

                criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
                optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12,
                                                      gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
                model.to(device)
                criterion.to(device)
                model_ft, hist, h, best_acc = train_model(model=model, optimizer=optimizer, scheduler=scheduler,extractor=extractor,
                                                          trainval_loaders=trainval_loaders, test_loaders=test_dataloader,early=early )

                plot_history(h, trial)
                test_model(model=model_ft, test_dataloader=test_dataloader,early=early, extractor=extractor, save=True)
                for i in range(2):
                 shutil.rmtree(os.path.join(Path.db_dir()[1][i]))
            write_log('\n' + 'Average and std of accuracy on test set')
            write_log('mean: {:.3f}, std:{:.4f}'.format(sum(acc_ls) / len(acc_ls), statistics.stdev(acc_ls)))
            write_log('\n' + 'Average and std of f1 score on test set')
            write_log('mean: {:.3f}, std:{:.4f}'.format(sum(f1score) / len(f1score), statistics.stdev(f1score)))
            trial = 0












