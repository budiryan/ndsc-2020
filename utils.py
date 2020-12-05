import numpy as np
import pandas as pd
import re
import textdistance
import time
import torch

from glob import glob
from torch import nn
from scipy.spatial import distance
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from lightgbm.sklearn import LGBMClassifier
from PIL import Image, ImageFile
from contextlib import contextmanager
from fuzzywuzzy import fuzz

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

default_stop_words = [
    'atau', 'dan', 'and', 'murah', 'grosir',
    'untuk', 'termurah', 'cod', 'terlaris', 'bisacod', 'terpopuler',
    'bisa', 'terbaru', 'tempat', 'populer', 'di', 'sale', 'bayar', 'flash',
    'promo', 'seler', 'in', 'salee', 'diskon', 'gila', 'starseller', 'seller'
]


@contextmanager
def timer(task_name="timer"):
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0)) \
        # def timer


# Getting Image Features

def open_image(fname, size=224):
    img = Image.open(fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    t = t.permute(2, 0, 1).float() / 255.0
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    t = preprocess(t)
    t = t.reshape((1, *t.shape)).cuda()
    return t


# def open_image

def torch_load_hub_model(model_name, cut=-1):
    model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    if cut is not None:
        modules = list(model.children())[:cut]
    else:
        modules = list(model.children())
    model = nn.Sequential(*modules)
    return model.cuda()


# def torch_load_hub_model

def get_image_feature_vectors(model, image_path, img_size=224, verbose=True):
    feature_vector_dict = {}
    i = 0
    for filename in glob(f'{image_path}/*'):
        if verbose:
            print(i, filename.split('/')[-1])
        i += 1
        img = open_image(filename, size=img_size)
        vect = model.forward(img)
        vect = vect.detach().cpu().numpy()
        vect = vect.reshape((vect.shape[1],))
        feature_vector_dict[filename.split('/')[-1]] = vect
    return feature_vector_dict


# def get_image_feature_vectors

def get_batch_image_feature_vectors(model, img_subdir_path, img_size=224, batch_size=100):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=img_subdir_path, transform=tfm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    i = 0
    img_feats = {}
    for i, data in enumerate(dataloader):
        img, labels = data
        v = model.forward(img.cuda()).cpu().detach().numpy()
        v = v.reshape((v.shape[0], v.shape[1]))
        i += i
        for i in range(v.shape[0]):
            fname = dataset.imgs[idx][0].split('/')[-1]
            img_feats[fname] = v[i]
            idx += 1

    return img_feats


def get_hist_feature_vectors(img_path, img_size=224, bins=15, verbose=True):
    im2hist = lambda x: np.histogram(x, bins=bins, range=(0, 1))
    hist_feats = {}
    idx = 0
    for filename in glob(f'{img_path}/*'):
        print(filename)
        data = np.array(Image.open(filename).resize((img_size, img_size))) / 255.0
        hists = []
        for i in range(3):
            hist, _ = im2hist(data[i])
            hists.append(hist / max(hist))
        hist_feats[filename.split('/')[-1]] = np.concatenate(hists)
        if verbose and idx + 1 % 10 == 0:
            print("Current index: %d" % idx)
        idx += 1
    return hist_feats


# def get_hist_feature_vectors

def build_hist_feats(df, hist_feats_dict):
    hist_1_feats = []
    hist_2_feats = []
    # generate columns and labels
    for i, row in df.iterrows():
        v1 = hist_feats_dict[row['image_1']]
        hist_1_feats.append(v1)
        v2 = hist_feats_dict[row['image_2']]
        hist_2_feats.append(v2)
    hist_1_feats = np.array(hist_1_feats)
    hist_2_feats = np.array(hist_2_feats)
    hist_feat_len = list(hist_feats_dict.values())[0].shape[0]
    col_list = [f'hist_1_{i}' for i in range(hist_feat_len)] + \
               [f'hist_2_{i}' for i in range(hist_feat_len)]
    feats_df = pd.DataFrame(
        data=np.concatenate([hist_1_feats, hist_2_feats], axis=1),
        columns=col_list
    )
    return feats_df


# def build_hist_feats


# Distance Feature Engineering
def gen_dist_feats(vect_img_1, vect_img_2):
    return [
        distance.euclidean(vect_img_1, vect_img_2),
        distance.braycurtis(vect_img_1, vect_img_2),
        distance.canberra(vect_img_1, vect_img_2),
        distance.chebyshev(vect_img_1, vect_img_2),
        distance.cityblock(vect_img_1, vect_img_2),
        distance.cosine(vect_img_1, vect_img_2),
        distance.jensenshannon(vect_img_1, vect_img_2),
        distance.minkowski(vect_img_1, vect_img_2),
        skew(np.nan_to_num(vect_img_1)),
        skew(np.nan_to_num(vect_img_2)),
        kurtosis(np.nan_to_num(vect_img_1)),
        kurtosis(np.nan_to_num(vect_img_2)),
    ]


# def gen_dist_feats
def build_img_feats(df, img_feats_dict, label_col=None):
    image_1_feats = []
    image_2_feats = []
    dist_feats = []
    labels = []
    # generate columns and labels
    for i, row in df.iterrows():
        if label_col is not None:
            labels.append(row[label_col])
        v1 = img_feats_dict[row['image_1']]
        image_1_feats.append(v1)
        v2 = img_feats_dict[row['image_2']]
        image_2_feats.append(v2)
        dist_feats.append(gen_dist_feats(v1, v2))
    image_1_feats = np.array(image_1_feats)
    image_2_feats = np.array(image_2_feats)
    dist_feats = np.array(dist_feats)
    labels = np.array(labels)
    # rename columns
    dist_lst = [
        'img_euclidean_dist',
        'img_braycurtis_dist',
        'img_canberra_dist',
        'img_chebyshev_dist',
        'img_cityblock_dist',
        'img_cosine_dist',
        'img_jensenshannon_dist',
        'img_minkowski_dist',
        'img_1_skew',
        'img_2_skew',
        'img_1_kurt',
        'img_2_kurt'
    ]
    img_feat_len = list(img_feats_dict.values())[0].shape[0]
    col_list = [f'img_1_{i}' for i in range(img_feat_len)] + \
               [f'img_2_{i}' for i in range(img_feat_len)] + dist_lst
    if label_col is not None:
        col_list.extend([label_col])
        feats_df = pd.DataFrame(
            data=np.concatenate(
                [image_1_feats, image_2_feats, dist_feats,
                 labels.reshape(-1, 1)], axis=1),
            columns=col_list
        )
        feats_df[label_col] = df[label_col].astype(int)
    else:
        feats_df = pd.DataFrame(
            data=np.concatenate(
                [image_1_feats, image_2_feats, dist_feats],
                axis=1),
            columns=col_list
        )

    return feats_df


# def build_img_feats

def preprocess_text(text):
    s = str(text).lower()
    # replace & with and
    s = re.sub('&', ' and ', s)
    # replace / with or (idn)
    s = re.sub('/', 'atau', s, count=1)
    # remove all special characters
    s = s = re.sub(r"[^a-zA-Z0-9]+", ' ', s)
    # replace 's with only s (the special character ' is not the standard one, hence the implementation)
    s = re.sub(' s ', 's ', s)
    # add whitespace after each number
    s = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", s).strip()
    return s


# def preprocess_text

def remove_stopwords(text):
    s = str(text).lower()
    s = " ".join([word for word in s.split() if word not in default_stop_words])
    return s


# def remove_stopwords

def preprocess_row(row, col, func):
    return func(row[col])


# def preprocess_row

def preprocess_text_df(df, txt_cols=['title_1', 'title_2'], func=preprocess_text):
    txt_df = df[txt_cols].copy()
    for col in txt_cols:
        txt_df[col] = txt_df.apply(lambda x: preprocess_row(x, col, func=func), axis=1)
    return txt_df


# def preprocess_text_df

def build_handcraft_text_feats(tmp_df):
    df = tmp_df.copy()
    df['len_title_1'] = df.title_1.apply(lambda x: len(str(x)))
    df['len_title_2'] = df.title_2.apply(lambda x: len(str(x)))
    df['diff_len'] = df.len_title_1 - df.len_title_2
    df['abs_diff_len'] = abs(df.len_title_1 - df.len_title_2)
    df['len_char_title_1'] = df.title_1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    df['len_char_title_2'] = df.title_2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    df['len_word_title_1'] = df.title_1.apply(lambda x: len(str(x).split()))
    df['len_word_title_2'] = df.title_2.apply(lambda x: len(str(x).split()))
    df['common_words'] = df.apply(
        lambda x: len(set(str(x['title_1']).lower().split()).intersection(set(str(x['title_2']).lower().split()))),
        axis=1)
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_partial_token_set_ratio'] = df.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_partial_token_sort_ratio'] = df.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['title_1']), str(x['title_2'])), axis=1)
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['title_1']), str(x['title_2'])),
                                           axis=1)
    df['txt_hamming'] = df.apply(
        lambda x: textdistance.hamming.normalized_similarity(str(x['title_1']), str(x['title_2'])), axis=1)
    df['txt_damerau_levenshtein'] = df.apply(
        lambda x: textdistance.damerau_levenshtein.normalized_similarity(str(x['title_1']), str(x['title_2'])), axis=1)
    df['txt_jaro_winkler'] = df.apply(
        lambda x: textdistance.jaro_winkler.normalized_similarity(str(x['title_1']), str(x['title_2'])), axis=1)
    df['txt_overlap'] = df.apply(
        lambda x: textdistance.overlap.normalized_similarity(str(x['title_1']), str(x['title_2'])), axis=1)
    df['txt_mra'] = df.apply(lambda x: textdistance.mra.normalized_similarity(str(x['title_1']), str(x['title_2'])),
                             axis=1)
    df.drop(columns=['title_1', 'title_2'], inplace=True)

    return df


# def build_handcraft_text_feats


def build_text_feats(df, model):
    title_1_feats = []
    title_2_feats = []
    dist_feats = []
    for i, row in df.iterrows():
        title_1_feats.append(model.get_sentence_vector(row['title_1']))
        title_2_feats.append(model.get_sentence_vector(row['title_2']))
        dist_feats.append(gen_dist_feats(title_1_feats[-1], title_2_feats[-1]))
    title_1_feats = np.array(title_1_feats)
    title_2_feats = np.array(title_2_feats)
    dist_feats = np.array(dist_feats)
    dist_lst = [
        'txt_euclidean_dist',
        'txt_braycurtis_dist',
        'txt_canberra_dist',
        'txt_chebyshev_dist',
        'txt_cityblock_dist',
        'txt_cosine_dist',
        'txt_jensenshannon_dist',
        'txt_minkowski_dist',
        'txt_1_skew',
        'txt_2_skew',
        'txt_1_kurt',
        'txt_2_kurt'
    ]
    txt_feat_len = title_1_feats.shape[1]
    handcraft_feats_df = build_handcraft_text_feats(df[['title_1', 'title_2']])
    col_list = [f'txt_1_{i}' for i in range(txt_feat_len)] + \
               [f'txt_2_{i}' for i in range(txt_feat_len)] + dist_lst \
               + handcraft_feats_df.columns.tolist()
    feats_df = pd.DataFrame(
        data=np.concatenate([title_1_feats, title_2_feats, dist_feats, handcraft_feats_df],
                            axis=1), columns=col_list
    )
    feats_df = feats_df.drop('txt_jensenshannon_dist', axis=1)
    print('Build handcraft text feats..')
    return feats_df


# def build_text_feats

# Model Train Script
def train_k_fold_lgbm(X, y, features, FOLDS=5, RANDOM_STATE=707, PARAM_COMBINATION=40):
    print(f'X shape: {X.shape}')

    lgbm_default = LGBMClassifier(learning_rate=0.1, n_estimators=450,
                                  max_depth=7, min_child_weight=1, subsample=0.8,
                                  class_weight='balanced', boosting='gbdt')
    lgbm_params = {
        'num_leaves': [6, 12, 24, 64],
        'max_depth': [3, 5, 7, 14],
        'min_data_in_leaf': [20, 40, 80],
        'min_sum_hessian_in_leaf': [1e-5, 1e-2, 1, 1e2, 1e4],
        'bagging_fraction': [i / 10.0 for i in range(7, 11)],
        'bagging_freq': [0, 5, 10, 20, 30],
        'feature_fraction': [i / 10.0 for i in range(3, 7)],
        'lambda_l1': [0, 1e-5, 1e-2],
        'lambda_l2': [0, 1e-5, 1e-2]
    }
    print('lgbm params: ', lgbm_params)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rd_lgbm = RandomizedSearchCV(
        estimator=lgbm_default,
        param_distributions=lgbm_params,
        scoring='f1_macro',
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        cv=skf.split(X.loc[:, features], y),
        verbose=1,
        random_state=RANDOM_STATE,
        n_iter=PARAM_COMBINATION
    )

    print(f'randomcv shape: {X.loc[:, features].shape}')

    rd_lgbm.fit(
        X=X.loc[:, features],
        y=y
    )

    feature_importance = pd.DataFrame(
        rd_lgbm.best_estimator_.feature_importances_,
        index=features,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    feature_hyped = feature_importance[feature_importance['importance'] > 0].index
    lgbm_hyped = LGBMClassifier(**rd_lgbm.best_estimator_.get_params())

    print('Training on whole population with best parameters and features...')
    final_features = list(feature_hyped)
    print(f'training final shape: {X.loc[:, final_features].shape}')
    lgbm_hyped.fit(
        X=X.loc[:, final_features],
        y=y
    )

    feature_importance = pd.DataFrame(
        lgbm_hyped.feature_importances_,
        index=final_features,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    print('Finished!')
    return feature_importance, final_features, lgbm_hyped

# train_k_fold_lgbm
