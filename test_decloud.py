import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from dataset_cloudy_binary import SEN12MS, ToTensor, Normalize
from models.HCR_LC_Net import HCR_LC_Net
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, \
    F2_score, Hamming_loss, Subset_accuracy, Accuracy_score, One_error, \
    Coverage_error, Ranking_loss, LabelAvgPrec_score, calssification_report, \
    conf_mat_nor, get_AA, multi_conf_mat, OA_multi, PSNR, SSIM, MAE, SAM
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score


label_choices = ['multi_label', 'single_label']

# ------------------------ define and parse arguments -------------------------
parser = argparse.ArgumentParser()

# configure
parser.add_argument('--exp_name', type=str, default="cloudy_multi_unetall_s1s2all",
                    help="experiment name. will be used in the path names \
                         for log- and savefiles. If no input experiment name, \
                         path would be set to model name.")
parser.add_argument('--config_file', type=str, default="/Volumes/GU/Masterthesis2021/models/cloudy_multi_unetall_s1s2all_uncertaintyloss_withpretext_rotation_3step_continue/logs/20211127_065203_arguments.txt",
                    help='path to config file')

# data directory
parser.add_argument('--data_dir', type=str, default="/Volumes/GU/Masterthesis2021/data/SEN12MS/m1554803/data",
                    help='path to SEN12MS dataset')
parser.add_argument('--label_split_dir', type=str, default="/Users/guziqi/Desktop/masterthesis/code/SEN12MS-master/label_split",
                    help="path to label data and split list")
parser.add_argument('--checkpoint_pth', type=str, default="/Volumes/GU/Masterthesis2021/models/cloudy_multi_unetall_s1s2all_uncertaintyloss_withpretext_rotation_3step_continue/checkpoints/20211105_170929_model_best.pth",
                    help='path to the pretrained weights file')

# hyperparameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='mini-batch size (default: 64)')
parser.add_argument('--num_workers',type=int, default=4,
                    help='num_workers for data loading in pytorch')

args = parser.parse_args()


# -------------------- set directory for saving files -------------------------
cloudy_dir = os.path.join('./', args.exp_name, 'cloudy/')
decloud_dir = os.path.join('./', args.exp_name, 'decloud/')
gt_dir = os.path.join('./', args.exp_name, 'groundtruth/')
mask_dir = os.path.join('./', args.exp_name, 'cloud_mask/')

if not os.path.isdir(cloudy_dir):
    os.makedirs(cloudy_dir)
if not os.path.isdir(decloud_dir):
    os.makedirs(decloud_dir)
if not os.path.isdir(gt_dir):
    os.makedirs(gt_dir)
if not os.path.isdir(mask_dir):
    os.makedirs(mask_dir)



# -------------------------------- Main Program ------------------------------
def main():
    global args

# -------------------------- load config from file
    # load config
    config_file = args.config_file
        
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            (key, val) = line.split()
            config[(key[0:-1])] = val
            
    # Convert string to boolean
    boo_use_s2 = config['use_s2'] == 'False'
    boo_use_s1 = config['use_s1'] == 'True'
    boo_use_RGB = config['use_RGB'] == 'False'
    boo_use_ALL = config['use_ALL'] == 'True'
    
    boo_IGBP_simple = config['IGBP_simple'] == 'True'
    
    # define label_type
    cf_label_type = config['label_type']
    assert cf_label_type in label_choices
    
    # define threshold 
    cf_threshold = float(config['threshold'])
    
    
    # define labels used in cls_report
    if boo_IGBP_simple:
        ORG_LABELS = ['1','2','3','4','5','6','7','8','9','10']
    else:
        ORG_LABELS = ['1','2','3','4','5','6','7','8','9','10',
                      '11','12','13','14','15','16','17']
    
    
# ----------------------------------- data
    # define mean/std of the training set (for data normalization)    
    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
                  
    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 
                
                
    # load test dataset
    imgTransform = transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)])
    
    test_dataGen = SEN12MS(args.data_dir, args.label_split_dir,
                           imgTransform = imgTransform,
                           label_type=cf_label_type, threshold=cf_threshold, subset="test", 
                           use_s1=boo_use_s1, use_s2=boo_use_s2, use_RGB=boo_use_RGB, use_ALL=boo_use_ALL,
                           #use_s1=True, use_s2=False, use_RGB=False, use_ALL=False,
                           IGBP_s=boo_IGBP_simple)
    
    # number of input channels
    n_inputs = test_dataGen.n_inputs
    print('input channels =', n_inputs)
    
    # set up dataloaders
    # num_workers=args.num_workers,
    test_data_loader = DataLoader(test_dataGen,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    
# -------------------------------- ML setup    
    # cuda
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    # define number of classes
    if boo_IGBP_simple:
        numCls = 10
    else:
        numCls = 17
        
    print('num_class: ', numCls)
    
    # define model
 
    model = HCR_LC_Net(n_inputs, numCls)
    
    
    # move model to GPU if is available
    if use_cuda:
        model = model.cuda()
        
    # import model weights
    checkpoint = torch.load(args.checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint_pth, checkpoint['epoch']))

    # set model to evaluation mode
    model.eval()

    # define metrics
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score() # from original script, not recommeded, seems not correct
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()
    
    calssification_report_ = calssification_report(ORG_LABELS)

    psnr_ = PSNR()
    ssim_ = SSIM()
    mae_ = MAE()
    sam_ = SAM()
    
# -------------------------------- prediction
    y_true = []
    predicted_probs = []

    psnr_all = []
    ssim_all = []
    mae_all = []
    sam_all = []
    sample_f1_per = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_data_loader, desc="test")):
            num = data["image"].size(0)
          
            # unpack sample
            bands = data["image"]
            img_clear = data["clear"]
            labels = data["label"]
            id = data["id"]
            cloud_mask = data["cloud_mask"]

            if batch_idx == 9000:
                print(id)

    
            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                #labels = labels.to(torch.device("cuda"))
            
            # forward pass 
            decloud, logits = model(bands)

            # convert logits to probabilies
            if cf_label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()
                  
            labels = labels.cpu().numpy() # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

            predicted_probs_per = list(probs)
            y_true_per = list(labels)

            predicted_probs_persample = np.asarray(predicted_probs_per)
            # convert predicted probabilities into one/multi-hot labels
            y_predicted_persample = (predicted_probs_persample >= 0.5).astype(np.float32)
            y_true_persample = np.asarray(y_true_per)

            macro_f1_per_, micro_f1_per_, sample_f1_per_ = f1_score_(y_predicted_persample, y_true_persample)
            sample_f1_per.append(micro_f1_per_)


            for i in range(num):

                #psnr_all[(i + 64 * batch_idx),:] = psnr_(img_clear[i,:,:,:].cpu().numpy(), decloud.cpu().numpy()[i,:,:,:])
                #ssim_all[(i + 64 * batch_idx),:] = ssim_(img_clear[i,:,:,:].cpu().numpy(), decloud.cpu().numpy()[i,:,:,:])
                psnr = psnr_(img_clear[i, :, :, :].cpu().numpy(), decloud.cpu().numpy()[i, :, :, :])
                ssim = ssim_(img_clear[i, :, :, :].cpu().numpy(), decloud.cpu().numpy()[i, :, :, :])
                psnr_all.append(psnr)
                ssim_all.append(ssim)
                
                mae = mae_(img_clear[i, :, :, :].cpu().numpy(), decloud.cpu().numpy()[i, :, :, :])
                sam = sam_(img_clear[i, :, :, :].cpu().numpy(), decloud.cpu().numpy()[i, :, :, :])
                mae_all.append(mae)
                sam_all.append(sam)

                # save cloudy images
                r_cloudy = ((np.array(bands[i, 3, :, :].cpu())))
                g_cloudy = ((np.array(bands[i, 2, :, :].cpu())))
                b_cloudy = ((np.array(bands[i, 1, :, :].cpu())))
                max_r = r_cloudy.max()
                max_g = g_cloudy.max()
                max_b = b_cloudy.max()
                min_r = r_cloudy.min()
                min_g = g_cloudy.min()
                min_b = b_cloudy.min()

                r_cloudy = (255 * (r_cloudy - min_r) / (max_r - min_r)).astype(int)
                g_cloudy = (255 * (g_cloudy - min_g) / (max_g - min_g)).astype(int)
                b_cloudy = (255 * (b_cloudy - min_b) / (max_b - min_b)).astype(int)

                rgb_cloudy = np.zeros((256, 256, 3))
                rgb_cloudy[:, :, 0] = r_cloudy
                rgb_cloudy[:, :, 1] = g_cloudy
                rgb_cloudy[:, :, 2] = b_cloudy

                rgb_cloudy = rgb_cloudy.astype(np.uint8)
                #plt.imshow(rgb_cloudy)

                plt.imsave(os.path.join(cloudy_dir, id[i] + 'f'), rgb_cloudy)

                #cv2.imwrite(str(), np.array())
                # save decloud images
                r_decloud = ((np.array(decloud[i, 3, :, :].cpu())))
                g_decloud = ((np.array(decloud[i, 2, :, :].cpu())))
                b_decloud = ((np.array(decloud[i, 1, :, :].cpu())))
                max_r = r_decloud.max()
                max_g = g_decloud.max()
                max_b = b_decloud.max()

                min_r = r_decloud.min()
                min_g = g_decloud.min()
                min_b = b_decloud.min()

                r_decloud = (255 * (r_decloud - min_r) / (max_r - min_r)).astype(int)
                g_decloud = (255 * (g_decloud - min_g) / (max_g - min_g)).astype(int)
                b_decloud = (255 * (b_decloud - min_b) / (max_b - min_b)).astype(int)

                rgb_decloud = np.zeros((256, 256, 3))
                rgb_decloud[:, :, 0] = r_decloud
                rgb_decloud[:, :, 1] = g_decloud
                rgb_decloud[:, :, 2] = b_decloud
                rgb_decloud = rgb_decloud.astype(np.uint8)
                plt.imsave(os.path.join(decloud_dir, id[i] + 'f'), rgb_decloud)

                # save ground truth images
                r_gt = ((np.array(img_clear[i, 3, :, :].cpu())))
                g_gt = ((np.array(img_clear[i, 2, :, :].cpu())))
                b_gt = ((np.array(img_clear[i, 1, :, :].cpu())))
                max_r = r_gt.max()
                max_g = g_gt.max()
                max_b = b_gt.max()

                min_r = r_gt.min()
                min_g = g_gt.min()
                min_b = b_gt.min()

                r_gt = (255 * (r_gt - min_r) / (max_r - min_r)).astype(int)
                g_gt = (255 * (g_gt - min_g) / (max_g - min_g)).astype(int)
                b_gt = (255 * (b_gt - min_b) / (max_b - min_b)).astype(int)

                rgb_gt = np.zeros((256, 256, 3))
                rgb_gt[:, :, 0] = r_gt
                rgb_gt[:, :, 1] = g_gt
                rgb_gt[:, :, 2] = b_gt
                rgb_gt = rgb_gt.astype(np.uint8)
                plt.imsave(os.path.join(gt_dir, id[i] + 'f'), rgb_gt)

                # save cloud mask
                mask = cloud_mask[i, :, :]
                plt.imsave(os.path.join(mask_dir, id[i] + 'f'), mask, cmap="gray")
            



    psnr_all = np.mean(psnr_all)
    ssim_all = np.mean(ssim_all)
    mae_all = np.mean(mae_all)
    sam_all = np.mean(sam_all)

    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels
    if cf_label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i, loc[i]] = 1

    y_true = np.asarray(y_true)

# --------------------------- evaluation with metrics  
    # general
    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)
    # ranking-based
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    cls_report = calssification_report_(y_predicted, y_true)
    
    
    if cf_label_type == 'multi_label':
        [conf_mat, cls_acc, aa] = multi_conf_mat(y_predicted, y_true, n_classes=numCls)
        # the results derived from multilabel confusion matrix are not recommended to use
        oa = OA_multi(y_predicted, y_true)
        # this oa can be Jaccard index 
        
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
            "clsReport": cls_report,
            "multilabel_conf_mat": conf_mat,
            "class-wise Acc": cls_acc,
            "AverageAcc": aa,
            "OverallAcc": oa,
            "PSNR": psnr_all,
            "SSIM": ssim_all,
            "MAE": mae_all,
            "SAM": sam_all,
             }
                
    else:
        conf_mat = conf_mat_nor(y_predicted, y_true, n_classes=numCls)
        #print (y_predicted, y_true)
        aa = get_AA(y_predicted, y_true, n_classes=numCls) # average accuracy, \
        # zero-sample classes are not excluded

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
            "clsReport": cls_report,
            "conf_mat": conf_mat,
            "AverageAcc": aa,
            "PSNR": psnr_all,
            "SSIM": ssim_all,
            "MAE": mae_all,
            "SAM": sam_all,
             }

    print("saving metrics...")
    print(micro_f1)
    print(micro_prec)
    print(micro_rec)
    print(psnr_all)
    print(ssim_all)
    print(mae_all)
    print(sam_all)
    pkl.dump(info, open("./test_scores_uncertainty.pkl", "wb"))
    pkl.dump(sample_f1_per, open("./f1_score_persample_13_test.pkl", "wb"))
    #pkl.dump(sample_f1, open("./f1_score_persample_new.pkl", "wb"))


if __name__ == "__main__":
    main()
