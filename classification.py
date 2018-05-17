# -*- coding: utf-8 -*-
import sys
import os
import gc
from argparse import ArgumentParser
from collections import Counter
import logging
import pandas as pd
import pickle
import h5py
import time
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.optimizers import Adam
# Google Client Libraries
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
# roc_auc_callback.py in the current directory
import roc_auc_callback as auc


client = language.LanguageServiceClient()

FTEXT_PRETRAINED_NEOLOGD_PATH = './fast_neo.vec'
FTEXT_PRETRAINED_WIKI_PATH = './fast_wiki.vec'
TOKENIZER_PATH = './tokenizer.pkl'
NEW_WORDS_LIST_PATH = './new_wordlist.pkl'
MODEL_HISTORY_PATH = './hist.pkl'
PRETRAINED_MODEL_PATH = './best_model_no_cross_val.h5'


def train(train_path):
    if os.path.exists(train_path):
        print("学習ファイルを読み込んでいます")
        train_df = pd.read_csv(train_path)
    else:
        sys.exit("学習ファイルが見つかりません")
    if "comment_text" in train_df.columns and "is_toxic" in train_df.columns:
        pass
    else:
        sys.exit("学習データ内のカラム名が正しくありません。入力テキストのカラム名を \
                 \"comment_text\"、\n 出力データのカラム名を\"is_toxic\"と指定してください。")
    
    seq_maxlen = 300
    embed_size = 300
    X_train_np, y_train_np, tokenizer = _get_train_feature(train_df, seq_maxlen=seq_maxlen)
    ftext_neo_wmatrix, ftext_wiki_wmatrix, unique_rate_np = _get_train_wvector_coeff(train_df, 
                                                                                     tokenizer, 
                                                                                     embed_size=embed_size,
                                                                                     seq_maxlen=seq_maxlen)
    X_train_np = np.concatenate([X_train_np, unique_rate_np], axis=1)
    del train_df
    gc.collect()

    seed = 223
    val_ratio = 0.05
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train_np,
                                                                  y_train_np, 
                                                                  random_state=seed, 
                                                                  test_size=val_ratio)
    len_train = len(X_train_np)

    X_train_dict = _get_input_dict(X_train_np, seq_maxlen)
    X_val_dict = _get_input_dict(X_val_np, seq_maxlen)
    del X_train_np, X_val_np
    gc.collect()

    batch_size = 5000
    epochs = 2
    num_words = min(tokenizer.num_words, len(tokenizer.word_index))
    model = _get_model(len_train, 
                       ftext_neo_wmatrix, 
                       ftext_wiki_wmatrix,
                       num_words=num_words,  
                       seq_maxlen=seq_maxlen,
                       embed_size=embed_size,
                       batch_size=batch_size,
                       epochs=epochs)
    false_cross_validation = 0
    auc_callback = auc.RocAucEvaluation(validation_data=(X_val_dict, y_val_np),
                                        fold_idx=false_cross_validation)
    history = model.fit(X_train_dict, 
                        y_train_np, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_val_dict, y_val_np),
                        callbacks=[auc_callback], 
                        class_weight={0:0.01, 1:.99}, 
                        shuffle=True, 
                        verbose=1)

    print("学習が完了しました。ファイル名：{:s}として履歴を保存しています。".format(MODEL_HISTORY_PATH))
    with open(MODEL_HISTORY_PATH, 'wb') as handle:
        pickle.dump(history.history, handle)

def pred(test_path):
    if os.path.exists(test_path):
        print("テストファイルを読み込んでいます")
        test_df = pd.read_csv(test_path)
    else:
        sys.exit("テストファイルが見つかりません")
    if "comment_text" in test_df.columns:
        pass
    else:
        sys.exit("テストファイルデータ内のカラム名が正しくありません。\
                 入力テキストのカラム名を\"comment_text\"と指定してください。")
    
    seq_maxlen = 300
    embed_size = 300
    X_test_np, tokenizer = _get_test_feature(test_df, seq_maxlen=seq_maxlen)
    unique_rate_np = _get_test_wvector_coeff(test_df, 
                                             tokenizer, 
                                             embed_size=embed_size,
                                             seq_maxlen=seq_maxlen)
    X_test_np = np.concatenate([X_test_np, unique_rate_np], axis=1)
    X_test_dict = _get_input_dict(X_test_np, seq_maxlen)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print("学習済みモデルを読み込んでいます")
        model = load_model(PRETRAINED_MODEL_PATH)
    else:
        sys.exit("学習済みモデルが読み込めません")
    batch_size=5000
    outputs = model.predict(X_test_dict, batch_size=batch_size, verbose=1)
    
    print("テストデータの推論が終了しました。ファイル名: output.csv として結果を保存しています。")
    test_df["is_toxic"] = outputs
    test_df.to_csv("output.csv", index=False)

def pred_text(input_text):
    seq_maxlen = 300
    embed_size = 300
    X_test_np, tokenizer = _get_test_feature(input_text, seq_maxlen=seq_maxlen)
    unique_rate_np = _get_test_wvector_coeff(input_text, 
                                             tokenizer, 
                                             embed_size=embed_size,
                                             seq_maxlen=seq_maxlen)
    X_test_np = np.concatenate([X_test_np, unique_rate_np], axis=1)
    X_test_dict = _get_input_dict(X_test_np, seq_maxlen)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print("学習済みモデルを読み込んでいます")
        model = load_model(PRETRAINED_MODEL_PATH)
    else:
        sys.exit("学習済みモデルが読み込めません")
    outputs = model.predict(X_test_dict, verbose=1)
    print("Toxicity of " + input_text + " is ", outputs[0, 0])

def _get_train_feature(train_df, seq_maxlen=300):
    """
    コメント(文字列) -> トークン化された特徴, ラベル
    """
    print("コメントを分かち書きしています")
    train_df["comment_text"] = train_df["comment_text"].apply(lambda text: _get_separeted(text))
    train_np = train_df["comment_text"].values

    print("学習データの語彙数を計算しています")
    total_word_count = Counter()
    for comment in train_df["comment_text"]: total_word_count += Counter(comment.split())
    num_words = len(total_word_count.keys())

    print("tokenizerモデルを学習しています")
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train_np))

    print("学習済みtokenizerを保存しています")
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("コメントをtokenizeしています")
    tokened_train_list = tokenizer.texts_to_sequences(train_np)
    X_train_np = sequence.pad_sequences(tokened_train_list, maxlen=seq_maxlen)
    y_train_np = train_df["is_toxic"].values
        
    return X_train_np, y_train_np, tokenizer

def _get_train_wvector_coeff(train_df, tokenizer, embed_size=300, seq_maxlen=300):
    """
    トークン化した各単語に対応する学習済みfastText, GloVeの重みを抽出。 
    また、学習済みfastText, GloVeに存在しない単語をnew_words_listとして抽出し、
    train_df中にどのくらい新語が含まれているか計算する。
    """
    print("tokenの重みを取得しています")
    ftext_neo_wmatrix, new_words_list = _get_weighted_matrix(tokenizer, 
                                                             FTEXT_PRETRAINED_NEOLOGD_PATH, 
                                                             embed_size=embed_size,
                                                             seq_maxlen=seq_maxlen)
    ftext_wiki_wmatrix, _  = _get_weighted_matrix(tokenizer, 
                                                  FTEXT_PRETRAINED_WIKI_PATH, 
                                                  embed_size=embed_size,
                                                  seq_maxlen=seq_maxlen)

    print("学習済みword vectorに含まれない新出語の比率を計算しています")
    train_df["word_list"] = train_df["comment_text"].apply(lambda comment: comment.split())
    # word_list中にnew_word_listの単語がどのくらい含まれているか数える
    unique_rate_np = train_df["word_list"].apply(lambda word_list: 
                                                 len([word for word in word_list if word in new_words_list])
                                                 / len(word_list)).astype("float16").values

    return ftext_neo_wmatrix, ftext_wiki_wmatrix, unique_rate_np.reshape(-1, 1)

def _get_separeted(text):
    """
    Natural Language APIを用いて，文章を単語ごとに分かち書きする．
    
    注：100 秒あたりのリクエスト数はデフォルトで1000が上限。制限に引っかからないようにすること.
    上限数を上げたい場合は https://console.cloud.google.com/project/_/quotas?_ga=2.241281344.-390384268.1526126587
    の IAMと管理 > 割り当て > Cloud Natural Language API Requests / 分 の割り当て制限量の増加をリクエストする．
    """
    document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
    syntax_response = client.analyze_syntax(document=document)
    separeted_text = " ".join([s.text.content for s in syntax_response.tokens])
    time.sleep(0.1)
    
    return separeted_text

def _get_weighted_matrix(tokenizer, pretrained_path, embed_size=300, seq_maxlen=300):
    """学習済み単語ベクトルを読み込んで、特徴の重みを計算"""
    embed_idx = dict(_get_coefs(*word_and_vector.strip().split()) for word_and_vector in open(pretrained_path, encoding="utf-8"))
    # 次元の不揃いなベクトルをseq_maxlenにpaddingする
    embed_idx_val = sequence.pad_sequences(embed_idx.values(), maxlen=seq_maxlen)
    all_embs = np.stack(embed_idx_val)
    embed_mean, embed_std =  all_embs.mean(), all_embs.std()
    word_idx = tokenizer.word_index
    nb_words = min(tokenizer.num_words, len(word_idx))
    embed_matrix = np.random.normal(embed_mean, embed_std, (nb_words, embed_size))
    new_words_list = []
    for word, idx in word_idx.items():
        if idx >= nb_words: continue
        embed_vector = embed_idx.get(word)
        if embed_vector is not None: embed_matrix[idx] = embed_vector
        else: new_words_list.append(word)

    return embed_matrix, new_words_list

def _get_coefs(word, *arr):
    """
    入力された*arr(str)にワードベクトルの数値以外の不純物が含まれていることがあるので，
    try: 数値データであればarr_fixedにappendする
    except: 数値以外の不純物が混じっている場合（floatへキャスト時にValueErrorする場合）は含めない
    """
    arr_fixed = []
    for arr_val in arr:
        try:
            arr_fixed.append(float(arr_val))
        except ValueError:
            pass

    return word, np.asarray(arr_fixed, dtype='float32')

def _get_input_dict(input_np, seq_maxlen=300):
    """2種類の学習済み分散表現モデル（fastText_NEologd, fastText_Wikipedia）で重み付けできるようにdictを作成"""
    input_dict = {
                  'ftext_neo': input_np[:, :seq_maxlen],
                  'ftext_wiki': input_np[:, :seq_maxlen],
                  'uniq_rate': input_np[:, seq_maxlen]
                 }

    return input_dict

def _get_model(len_train, ftext_neo_weight, ftext_wiki_weight, num_words, seq_maxlen=300, 
               embed_size=300, batch_size=5000, epochs=2):
    inp_ftext_neo = Input(shape=(seq_maxlen, ), name='ftext_neo')
    inp_ftext_wiki = Input(shape=(seq_maxlen, ), name='ftext_wiki')
    inp_urate = Input(shape=[1], name='uniq_rate')
    embed_ftext_neo = Embedding(num_words, embed_size, weights=[ftext_neo_weight])(inp_ftext_neo)
    embed_ftext_wiki = Embedding(num_words, embed_size, weights=[ftext_wiki_weight])(inp_ftext_wiki)
    conc_embed = concatenate([embed_ftext_neo, embed_ftext_wiki])
    dout = SpatialDropout1D(0.5)(conc_embed)

    lstmed = Bidirectional(CuDNNLSTM(40, return_sequences=True, go_backwards=True))(dout)
    grued = Bidirectional(CuDNNGRU(40, return_sequences=True, go_backwards=True))(lstmed)
    avg_pooled = GlobalAveragePooling1D()(grued)
    max_pooled = GlobalMaxPooling1D()(grued)
    conc_pool_urate = concatenate([avg_pooled, max_pooled, inp_urate])
    outs = Dense(1, activation="sigmoid")(conc_pool_urate)

    inputs = [inp_ftext_neo, inp_ftext_wiki, inp_urate]
    model = Model(inputs=inputs, outputs=outs)

    # 重み減数の設定
    steps = int(len_train / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = _get_exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer_adam,
                  metrics=['accuracy'])

    return model

def _get_exp_decay(init, fin, steps): return (init / fin) ** (1 / (steps - 1)) - 1 

def _get_test_feature(test_data, seq_maxlen=300):
    print("コメントを分かち書きしています")
    if type(test_data) is not str:
        # DataFrame -> numpy.array
        test_data["comment_text"] = test_data["comment_text"].apply(lambda text: _get_separeted(text))
        test_np = test_data["comment_text"].values
    else:
        # string -> numpy.array
        separeted_text = _get_separeted(test_data)
        test_np = np.array([separeted_text])
    if os.path.exists(TOKENIZER_PATH):
        print("学習済みtokenizerをロードしています")
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        sys.exit("学習済みtokenizerが見つかりません")
    
    tokened_test_list = tokenizer.texts_to_sequences(test_np)
    X_test_np = sequence.pad_sequences(tokened_test_list, maxlen=seq_maxlen)
        
    return X_test_np, tokenizer

def _get_test_wvector_coeff(test_data, tokenizer, embed_size=300, seq_maxlen=300):
    if os.path.exists(NEW_WORDS_LIST_PATH):
        with open(NEW_WORDS_LIST_PATH, 'rb') as handle:
            new_words_list = pickle.load(handle)
    else:
        print("new words listが存在しないので、作成します")
        _, new_words_list = _get_weighted_matrix(tokenizer, 
                                                 FTEXT_PRETRAINED_NEOLOGD_PATH, 
                                                 embed_size=embed_size,
                                                 seq_maxlen=seq_maxlen)
    if type(test_data) is not str:
        unique_rate_np = _get_urate_from_df(test_data, new_words_list)
    else:
        unique_rate_np = _get_urate_from_text(test_data, new_words_list)

    return unique_rate_np.reshape(-1, 1)

def _get_urate_from_df(test_data, new_words_list):
    """DataFrame形式のテストデータ -> word_list中にnew_word_listの単語が含まれている比率"""
    test_data["word_list"] = test_data["comment_text"].apply(lambda comment: comment.split())
    unique_rate_np = test_data["word_list"].apply(lambda word_list: 
                                                  len([word for word in word_list if word in new_words_list])
                                                  / len(word_list)).astype("float16").values

    return unique_rate_np

def _get_urate_from_text(test_data, new_words_list):
    """文字列(str)のテストデータ -> word_list中にnew_word_listの単語が含まれている比率"""
    word_list = test_data.split()
    unique_rate = len([word for word in word_list if word in new_words_list]) / len(word_list)

    return np.array([unique_rate])

def parser():
    argparser = ArgumentParser()
    argparser.add_argument('--train', type=str, nargs=1)
    argparser.add_argument('--pred', type=str, nargs=1)
    argparser.add_argument('--pred-text', type=str, nargs=1, dest='pred_text')
    args = argparser.parse_args()
    if args.train:
        return train(args.train[0])
    if args.pred:
        return pred(args.pred[0])
    if args.pred_text:
        return pred_text(args.pred_text[0])

if __name__ == '__main__':
    parser()