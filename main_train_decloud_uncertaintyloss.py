# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/

import os
import argparse
import numpy as np
from datetime import datetime 
from tqdm import tqdm

import torch
import torch.optim as optim 
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from s2cloudless import S2PixelCloudDetector
import torch.nn as nn
import torch.nn.functional as func

import shutil 
import sys
sys.path.append('../')

from dataset_cloudy_binary import SEN12MS, ToTensor, Normalize
from models.HCR_LC_Net import HCR_LC_Net, count_param
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, \
    F2_score, Hamming_loss, Subset_accuracy, Accuracy_score, One_error, \
    Coverage_error, Ranking_loss, LabelAvgPrec_score, PSNR, SSIM

from tensorboardX import SummaryWriter

model_choices = ['HCR_LC_Net', 'HCR']
label_choices = ['multi_label', 'single_label']

# ----------------------- define and parse arguments --------------------------
parser = argparse.ArgumentParser()

# experiment name
parser.add_argument('--exp_name', type=str, default="cloudy_unetall_s1s2all",
                    help="experiment name. will be used in the path names \
                         for log- and savefiles. If no input experiment name, \
                         path would be set to model name.")

# data directory
parser.add_argument('--data_dir', type=str, default=" ",
                    help='path to SEN12MS dataset')
parser.add_argument('--label_split_dir', type=str, default=" ",
                    help="path to label data and split list")

# input/output
parser.add_argument('--use_s2', action='store_true', default=False,
                    help='use sentinel-2 bands')
parser.add_argument('--use_s1', action='store_true', default=False,
                    help='use sentinel-1 data')
parser.add_argument('--use_RGB', action='store_true', default=False,
                    help='use sentinel-2 RGB bands')
parser.add_argument('--use_ALL', action='store_true', default=False,
                    help='use sentinel-2 ALL bands')
parser.add_argument('--IGBP_simple', action='store_true', default=True,
                    help='use IGBP simplified scheme; otherwise: IGBP original scheme')
parser.add_argument('--label_type', type=str, choices = label_choices,
                    default='multi_label',
                    help="label-type (default: multi_label)")
parser.add_argument('--threshold', type=float, default=0.1, 
                    help='threshold to convert probability-labels to multi-hot \
                    labels, mean/std for normalizatin would not be accurate \
                    if the threshold is larger than 0.22. \
                    for single_label threshold would be ignored')

# network
parser.add_argument('--model', type=str, choices = model_choices,
                    default='HCR_LC_Net',
                    help="network architecture (default: HCR_LC_Net)")

# training hyperparameters
parser.add_argument('--lr', type=float, default=0.001, 
                    help='initial learning rate')
parser.add_argument('--decay', type=float, default=1e-5, 
                    help='decay rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='mini-batch size (default: 64)')
parser.add_argument('--num_workers',type=int, default=0,
                    help='num_workers for data loading in pytorch')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs (default: 100)')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path to the pretrained weights file', )

args = parser.parse_args()

# -------------------- set directory for saving files -------------------------
if args.exp_name:
    checkpoint_dir = os.path.join('./', args.exp_name, 'checkpoints')
    logs_dir = os.path.join('./', args.exp_name, 'logs')
else:
    checkpoint_dir = os.path.join('./', args.model, 'checkpoints')
    logs_dir = os.path.join('./', args.model, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

# ----------------------------- saving files ---------------------------------
def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name +
                                               '_model_best.pth'))

#def save_checkpoint(state,name):

    #filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth')

    #torch.save(state, filename)


class CARLLossFunc(nn.Module):
    def __int__(self):
        super().__int__()

    def forward(self, input_cloudy, predicted, clear, cloud_mask):
        #cloud_mask = torch.from_numpy(cloud_mask)
        one = torch.ones_like(cloud_mask)
        #carl_error = torch.mean(
            #(one - cloud_mask) * torch.abs(predicted - input_cloudy) + cloud_mask * torch.abs(
                #predicted - clear)) + 1.0 * torch.mean(torch.abs(predicted - clear))
        carl_error = torch.mean(
            (torch.sub(one, cloud_mask)) * torch.abs(torch.sub(predicted,input_cloudy)) + torch.mul(cloud_mask, torch.abs(
                torch.sub(predicted, clear)))) +  torch.mul(1.0, torch.mean(torch.abs(torch.sub(predicted, clear))))
        return carl_error



# for uncertainty loss

# parameters for the weights
log_var_a = torch.zeros((1,), requires_grad=True)
log_var_b = torch.zeros((1,), requires_grad=True)


def uncertainty_loss(loss1, loss2, log_var_a, log_var_b):
    #precision_1 = torch.exp(-log_var_a)
    #precision_2 = 0.5 * torch.exp(-log_var_b)

    log_var_a = log_var_a.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    log_var_b = log_var_b.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    precision1 = torch.exp(-log_var_a)
    precision1 = precision1.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    loss = torch.sum(precision1 * loss1 + 0.5 * log_var_a, -1)

    precision2 = 0.5 * torch.exp(-log_var_b)
    precision2 = precision2.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    loss += torch.sum(precision2 * loss2 + 0.5 * log_var_b, -1)

    loss = torch.mean(loss)

    return loss


# -------------------------------- Main Program -------------------------------
def main():
    global args
    
    # save configuration to file
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    write_arguments_to_file(args, os.path.join(logs_dir, sv_name+'_arguments.txt'))

# ----------------------------------- data
    # define mean/std of the training set (for data normalization)
    label_type = args.label_type
        
    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
                  
    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 

    
    # load datasets 
    imgTransform = transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)])
    
    train_dataGen = SEN12MS(args.data_dir, args.label_split_dir, 
                            imgTransform=imgTransform, 
                            label_type=label_type, threshold=args.threshold, subset="train",
                            use_s1=args.use_s1, use_s2=args.use_s2, use_RGB=args.use_RGB, use_ALL=args.use_ALL,
                            IGBP_s=args.IGBP_simple)
    
    val_dataGen = SEN12MS(args.data_dir, args.label_split_dir, 
                          imgTransform=imgTransform, 
                          label_type=label_type, threshold=args.threshold, subset="val", 
                          use_s1=args.use_s1, use_s2=args.use_s2, use_RGB=args.use_RGB, use_ALL=args.use_ALL,
                          IGBP_s=args.IGBP_simple)
    
    # number of input channels
    n_inputs = train_dataGen.n_inputs 
    print('input channels =', n_inputs)
    
    # set up dataloaders
    train_data_loader = DataLoader(train_dataGen, 
                                   batch_size=args.batch_size, 
                                   num_workers=args.num_workers, 
                                   shuffle=True, 
                                   pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 shuffle=False, 
                                 pin_memory=True)

# -------------------------------- ML setup
    # cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    # define number of classes
    if args.IGBP_simple:
        numCls = 10
    else:
        numCls = 17
    
    print('num_class: ', numCls)
    
    # define model
    
    model = HCR_LC_Net(n_inputs, numCls)

    num_par = count_param(model)
    print(num_par)

    # move model to GPU if is available
    if use_cuda:
        model = model.cuda() 

    # define loss function
    if label_type == 'multi_label':
        lossfunc = torch.nn.BCEWithLogitsLoss()
    else:
        lossfunc = torch.nn.CrossEntropyLoss()

    lossfunc2 = CARLLossFunc()
    # set up optimizer
    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b])
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)

    best_acc = 0
    start_epoch = 0


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # set up tensorboard logging
    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))


# ----------------------------- executing Train/Val. 
    # train network
    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        name = sv_name + '_' + str(epoch)

        train(train_data_loader, model, optimizer, lossfunc, lossfunc2, label_type, epoch, use_cuda, train_writer)
        micro_f1 = val(val_data_loader, model, optimizer, label_type, epoch, use_cuda, val_writer)

        is_best_acc = micro_f1 > best_acc
        best_acc = max(best_acc, micro_f1)

        save_checkpoint({
            'epoch': epoch,
            'arch': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec': best_acc
            }, is_best_acc, sv_name)


def train(trainloader, model, optimizer, lossfunc, lossfunc2, label_type, epoch, use_cuda, train_writer):

    lossTracker = MetricTracker()
    
    # set model to train mode
    model.train()

    # main training loop
    for idx, data in enumerate(tqdm(trainloader, desc="training")):
        
        numSample = data["image"].size(0)
        
        # unpack sample
        bands = data["image"]
        clear_img = data["clear"]
        cloud_mask = data["cloud_mask"]
        cloud_mask = np.expand_dims(cloud_mask, 1).repeat(13, axis=1)

        if label_type == 'multi_label':
            labels = data["label"]
        else:
            labels = (torch.max(data["label"], 1)[1]).type(torch.long)
               
        # move data to gpu if model is on gpu
        if use_cuda:
            bands = bands.to(torch.device("cuda"))

            labels = labels.to(torch.device("cuda"))
            clear_img = clear_img.to(torch.device("cuda"))
            #cloud_mask = cloud_mask.to(torch.device("cuda"))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_mask = torch.tensor(cloud_mask, device=device).float()

        # reset gradients
        optimizer.zero_grad()
        
        # forward pass
        decloud, logits = model(bands)

        loss1 = lossfunc(logits, labels)
        loss2 = lossfunc2(bands[:,0:13,:,:], decloud, clear_img, cloud_mask)

        loss = uncertainty_loss(loss1, loss2, log_var_a, log_var_b)
        #loss = torch.relu(loss)

        # backward pass
        loss.backward()
        optimizer.step()

        lossTracker.update(loss.item(), numSample)


        std_1 = torch.exp(log_var_a) ** 0.5
        std_2 = torch.exp(log_var_b) ** 0.5

        std_1_ = std_1.item()
        std_2_ = std_2.item()

        weight1 = torch.exp(-log_var_a)
        weight2 = torch.exp(-log_var_b) * 0.5


        if idx % 50 == 0:  # every 1000 mini-batches...

            # ...log the running loss
            train_writer.add_scalar('loss/loss1', loss1.item(), epoch * len(trainloader) + idx)
            train_writer.add_scalar('loss/loss2', loss2.item(), epoch * len(trainloader) + idx)
            train_writer.add_scalar('loss/combined_loss', loss.item(), epoch * len(trainloader) + idx)
            train_writer.add_scalar('std/std1', std_1_, epoch * len(trainloader) + idx)
            train_writer.add_scalar('std/std2', std_2_, epoch * len(trainloader) + idx)
            train_writer.add_scalar('weight/loss1_weight', weight1.item(), epoch * len(trainloader) + idx)
            train_writer.add_scalar('weight/loss2_weight', weight2.item(), epoch * len(trainloader) + idx)


    #train_writer.add_scalar("loss", lossTracker.avg, epoch)


    print('Train loss: {:.6f}'.format(lossTracker.avg))

    print('weight1: {:.6f}'.format(weight1.item()))
    print('weight2: {:.6f}'.format(weight2.item()))

def val(valloader, model, optimizer, label_type, epoch, use_cuda, val_writer):

    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()
    psnr_ = PSNR()
    ssim_ = SSIM()


    # set model to evaluation mode
    model.eval()


    # main validation loop
    y_true = []
    predicted_probs = []
    psnr_all = []
    ssim_all = []


    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            # unpack sample
            bands = data["image"]
            img_clear = data["clear"]
            labels = data["label"]

            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                #labels = labels.to(torch.device("cuda"))

            # forward pass
            decloud, logits = model(bands)

            # convert logits to probabilies
            if label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()

            labels = labels.cpu().numpy() # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

            for i in range(data["image"].size(0)):
                psnr = psnr_(img_clear[i,:,:,:].cpu().numpy(), decloud.cpu().numpy()[i,:,:,:])
                ssim = ssim_(img_clear[i,:,:,:].cpu().numpy(), decloud.cpu().numpy()[i,:,:,:])
                psnr_all.append(psnr)
                ssim_all.append(ssim)

    psnr_all = np.mean(psnr_all)
    ssim_all = np.mean(ssim_all)


    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels
    if label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i,loc[i]] = 1

    y_true = np.asarray(y_true)

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)

    # Note that below 4 ranking-based metrics are not applicable to single-label
    # (multi-class) classification, but they will still show the scores during
    # validation on tensorboard
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "macroF2" : macro_f2,
            "microF2" : micro_f2,
            "sampleF2" : sample_f2,
            "HammingLoss" : hamming_loss,
            "subsetAcc" : subset_acc,
            "macroAcc" : macro_acc,
            "microAcc" : micro_acc,
            "sampleAcc" : sample_acc,
            "oneError" : one_error,
            "coverageError" : coverage_error,
            "rankLoss" : rank_loss,
            "labelAvgPrec" : labelAvgPrec,
            "PSNR": psnr_all,
            "SSIM": ssim_all
            }

    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Validation microPrec: {:.6f} microF1: {:.6f} sampleF1: {:.6f} microF2: {:.6f} sampleF2: {:.6f} PSNR: {:.6f} SSIM: {:.6f}' .format(
            micro_prec,
            micro_f1,
            sample_f1,
            micro_f2,
            sample_f2,
            psnr_all,
            ssim_all
            ))
    return micro_f1


if __name__ == "__main__":
    main()
