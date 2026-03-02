import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from shap import GradientExplainer



class SpectralDatasetPairedTrain(Dataset):
    """
    Input array in shape (num_pair, 2, num_features),
    extract paired samples. Label 1: same class; 0: diff class
    """
    def __init__(self, X_fn, y_fn):
        self.X = X_fn
        self.y = y_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        paired_x, y = self.X[index], self.y[index]
        qry = np.expand_dims(paired_x[0], axis=0)
        ref = np.expand_dims(paired_x[1], axis=0)
        qry = torch.from_numpy(qry).float()
        ref = torch.from_numpy(ref).float()
        return (qry, ref, y)


class SpectralDatasetPairedTest(Dataset):
    """
    Input array in shape (num_pair, 2, num_features),
    extract paired samples. No label in prediction.
    """
    def __init__(self, X_fn):
        self.X = X_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        paired_x = self.X[index]
        qry = np.expand_dims(paired_x[0], axis=0)
        ref = np.expand_dims(paired_x[1], axis=0)
        qry = torch.from_numpy(qry).float()
        ref = torch.from_numpy(ref).float()
        return (qry, ref)


class SpectralDatasetSingle(Dataset):
    """
    Input array in shape (num_spectra, num_features).
    Used for creating or matching against ref database. 
    """
    def __init__(self, X_fn):
        self.X = X_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class BranchWrapper(torch.nn.Module):
    def __init__(self, branch_model, ref_feat):
        super().__init__()
        self.branch = branch_model.eval()
        self.ref_feat = ref_feat

    def forward(self, x):
        feat = self.branch.forward_once(x)
        return F.pairwise_distance(feat, self.ref_feat.expand_as(feat), keepdim=True)


def load_feature_db(feat_db_path):
    db = h5py.File(feat_db_path, 'r')
    db = db['features']
    db = np.array(db)
    db = torch.tensor(db)
    return db


def save_feature_db(feat_db_path, features, tensor=False):
    if tensor:
        features = features.numpy()
    with h5py.File(feat_db_path, 'w') as f:
        f.create_dataset('features', data=features)


def input_files(filenames):
    """
    Full path of a file containing input numpy array,
    one filename per line. Only used in training.
    Input: _X.npy; Output: list of X.npy and y.npy
    """
    x_filenames = open(filenames).read().strip().split('\n')
    #y_filenames = [x.replace('_X.npy', '_y.npy') for x in x_filenames]
    #return x_filenames, y_filenames
    return x_filenames


def spectral_dataloader(X_fn, y_fn=None, batch_size=512, 
    num_workers=8, train=True, single=False, evaluate=False):
    if train:
        dataset = SpectralDatasetPairedTrain(X_fn, y_fn)
        if evaluate:
            shuffle = False
        else:
            shuffle = True
    else:
        if single:
            dataset = SpectralDatasetSingle(X_fn)
        else:
            dataset = SpectralDatasetPairedTest(X_fn)
        shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def run_train_and_eval(model, dataloader, cuda, training=True, optimizer=None, margin=1.0):
    if training:
        model.train()
    else:
        model.eval()
    total = 0
    for batch_idx, (qry, ref, label) in enumerate(dataloader):
        if cuda: 
            qry, ref, label = qry.cuda(), ref.cuda(), label.cuda()
        #label = label.unsqueeze(1)
        out_qry, out_ref = model.forward(qry, ref)
        loss = ContrastiveLoss(margin=margin)(out_qry, out_ref, label)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_value = loss.item()
        total += qry.size(0)
        
        if training:
            print(f'Training: batch_idx {batch_idx}, loss is {loss_value}, {total} sample pairs processed', flush=True)
        else:
            print(f'Evaluating: batch_idx {batch_idx}, loss is {loss_value}, {total} sample pairs processed', flush=True)


def create_reference_db(model, dataloader, cuda):
    feature_map = []
    model.eval()
    for batch_idx, inputs in enumerate(dataloader):
        if cuda:
            inputs = inputs.cuda()
        outputs = model.forward_create_database(inputs)
        if cuda:
            outputs = outputs.cpu()
        feature_map.append(outputs.detach().numpy())
    feature_map = np.concatenate(feature_map, axis=0) 
    return feature_map


def match_aginst_reference_twins(model, dataloader, cuda, feat_db):
    distance = []
    model.eval()
    for batch_idx, qry in enumerate(dataloader):
        if cuda:
            qry = qry.cuda()
            feat_db = feat_db.cuda()
        logit_array = model.forward_matching_twins(qry, feat_db)
        if cuda:
            logit_array = logit_array.cpu()
        logit_array = logit_array.detach().numpy()
        distance.append(logit_array)
    distance = np.concatenate(distance, axis=0)
    #probs = 1 / (1 + np.exp(-distance))
    return distance


# Save the checkpoint
def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)



def wrap_train_subfiles(fn, batch_size, model, cuda, optimizer, ckpt, margin=1.0, input_dir=None):
    """
    Train: input is given as txt with X.npy filenames, one per line.
    """
    x_filenames = input_files(fn)
    for f in x_filenames:
        print(f'Start training on {f}', flush=True)
        if input_dir is not None:
            f = os.path.join(input_dir, f)
        X = np.load(f)
        assert X.shape[1] == 2, "Training set should be paired."
        y = np.load(f.replace('_X.npy', '_y.npy'))
        dataloader = spectral_dataloader(X, y, batch_size=batch_size, train=True, single=False)
        run_train_and_eval(model, dataloader, cuda, training=True, optimizer=optimizer, margin=margin)
        if ckpt is not None:
            file_prefix = f.split('/')[-1]
            save_checkpoint(model, optimizer, ckpt + '_' + file_prefix.replace('_X.npy', '') + '.pth')


def wrap_generate_feature_map(fn, batch_size, model, cuda, weight, out_prefix, input_dir=None):
    if cuda:
        checkpoint = torch.load(weight)
    else:
        checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if input_dir is not None:
        fn = os.path.join(input_dir, fn)
    X = np.load(fn)
    assert len(X.shape) == 2, "Samples should be single for extracting feature map."
    dataloader = spectral_dataloader(X, batch_size=batch_size, train=False, single=True)
    features = create_reference_db(model, dataloader, cuda)
    save_feature_db(feat_db_path=out_prefix+'_features.h5', features=features)
    np.savetxt(out_prefix+'_features.txt', features, delimiter='\t')


def wrap_matcher(fn, batch_size, model, cuda, weight, out_prefix, feat_db, input_dir=None):
    db = load_feature_db(feat_db)
    if cuda:
        checkpoint = torch.load(weight)
    else:
        checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if input_dir is not None:
        fn = os.path.join(input_dir, fn)
    X = np.load(fn)
    #print(X.shape)
    assert len(X.shape) == 2, "Samples should be single for matching against ref db."
    dataloader = spectral_dataloader(X, batch_size=batch_size, train=False, single=True)
    dist = match_aginst_reference_twins(model, dataloader, cuda, db)
    np.savetxt(f"{out_prefix}_dist.txt", dist, delimiter='\t', fmt='%.4f') 



def elastic_net(fn_pos, fn_neg, output_model):
    pos_array = np.genfromtxt(fn_pos, delimiter='\t')
    pos_label = np.full(len(pos_array), 1, dtype=int)
    neg_array = np.genfromtxt(fn_neg, delimiter='\t')
    neg_label = np.full(len(neg_array), 0, dtype=int)
    
    X = np.concatenate([pos_array, neg_array], axis=0)
    y = np.concatenate([pos_label, neg_label], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    y_pred = (predictions[:, 1] > 0.5).astype(int)  # Convert predicted probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("EN Accuracy:", accuracy)
    auc = roc_auc_score(y_test, predictions[:, 1])
    print("EN AUC:", auc)
    joblib.dump(model, output_model)
    df = pd.DataFrame({'true': y_test, 'pred_prob': predictions[:, 1]})
    df.to_csv(output_model.replace('.pkl', '')+'-pred.csv', index=False)


def elastic_net_pred(model_pkl, dist_txt, output_csv):
    outside_array = np.genfromtxt(dist_txt, delimiter='\t')
    model = joblib.load(model_pkl)
    predictions = model.predict_proba(outside_array)
    df = pd.DataFrame({'pred_prob': predictions[:, 1]})
    df.to_csv(output_csv, index=False)



def shap_analysis(bg, train, test, model, weight, out_prefix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bg_data = np.load(bg)
    train_data = np.load(train)
    test_data = np.load(test)
    checkpoint = torch.load(weight, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    ref_idx = np.random.choice(len(train_data), min(len(train_data), 100), replace=False)
    
    with torch.no_grad():
        ref_m = torch.tensor(train_data[ref_idx, :10][:, None, :], dtype=torch.float32).to(device)
        ref_s = torch.tensor(train_data[ref_idx, 10:][:, None, :], dtype=torch.float32).to(device)
        centroid_m = model.model_morphol.forward_once(ref_m).mean(0, keepdim=True)
        centroid_s = model.model_spectra.forward_once(ref_s).mean(0, keepdim=True)

    bg_m = torch.tensor(bg_data[:, :10][:, None, :], dtype=torch.float32).to(device)
    bg_s = torch.tensor(bg_data[:, 10:][:, None, :], dtype=torch.float32).to(device)
    explainer_m = GradientExplainer(BranchWrapper(model.model_morphol, centroid_m), bg_m)
    explainer_s = GradientExplainer(BranchWrapper(model.model_spectra, centroid_s), bg_s)

    test_idx = np.random.choice(len(test_data), min(len(test_data), 100), replace=False)
    test_samples = test_data[test_idx]

    summary_list = []
    full_feature_list = []

    for i, sample in enumerate(test_samples):
        print(f"[{out_prefix}] analyzing {i+1}/{len(test_samples)}...")
        q_m = torch.tensor(sample[:10][None, None, :], dtype=torch.float32).to(device)
        q_s = torch.tensor(sample[10:][None, None, :], dtype=torch.float32).to(device)

        sh_m = explainer_m.shap_values(q_m, nsamples=100).squeeze() # (10,)
        sh_s = explainer_s.shap_values(q_s, nsamples=100).squeeze() # (2401,)

        summary_list.append({
            'sample_id': i,
            'morph_abs_sum': np.abs(sh_m).sum(),
            'morph_abs_mean': np.abs(sh_m).mean(),
            'spec_abs_sum': np.abs(sh_s).sum(),
            'spec_abs_mean': np.abs(sh_s).mean(),
            'density_ratio': np.abs(sh_m).mean() / (np.abs(sh_s).mean() + 1e-9)
        })

        full_row = {'sample_id': i}
        full_row.update({f'm_{j}': val for j, val in enumerate(sh_m)})
        full_row.update({f's_{j}': val for j, val in enumerate(sh_s)})
        full_feature_list.append(full_row)

    pd.DataFrame(summary_list).to_csv(f"{out_prefix}_summary.csv", index=False)
    pd.DataFrame(full_feature_list).to_csv(f"{out_prefix}_full_features.csv", index=False)
