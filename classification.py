# -*- coding: utf-8 -*-
import sys
import os
import gc
import logging
import pandas as pd
import pickle
import h5py
import sklearn.model_selection import train_test_split
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

FTEXT_PRETRAINED_PATH = './fast_blog.txt'
GLV_PRETRAINED_PATH = './glv_blog.txt'
TOKENIZER_PATH = './tokenizer.pkl'
FTEXT_WEIGHTED_MATRIX_PATH ='./fast_wmatrix.pkl' 
GLV_WEIGHTED_MATRIX_PATH ='./glv_wmatrix.pkl' 
NEW_WORDS_LIST_PATH = './new_wordlist.pkl'
MODEL_HISTORY_PATH = './train_hist/hist.pkl'
PRETRAINED_MODEL_PATH = './best_model_no_cross_val.h5'


def train(train_path):
    try:
        print("学習ファイルを読み込んでいます")
        train_df = pd.read_csv(train_path, usecols=["comment_text", "is_toxic"])
    except FileNotFoundError:
        sys.exit("学習ファイルが見つかりません")
    except ValueError:
        sys.exit("学習データ内のカラム名が正しくありません。入力テキストのカラム名を\"comment_text\"、\n 出力データのカラム名を\"is_toxic\"と指定してください。")
    
    seq_maxlen = 100
    num_words = 20000
    embed_size=128
    X_train_np, y_train_np, tokenizer = _get_train_feature(train_df, 
                                                           seq_maxlen=seq_maxlen, 
                                                           num_words=num_words)
    ftext_wmatrix, glv_wmatrix, unique_rate_np = _get_train_wvector_coeff(train_df, 
                                                                          tokenizer, 
                                                                          num_words=num_words,
                                                                          embed_size=embed_size)
    X_train_np = np.concatenate([X_train_np, unique_rate], axis=1)
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
    model = _get_model(len_train, 
                       ftext_wmatrix, 
                       glv_wmatrix, 
                       seq_maxlen=seq_maxlen, 
                       num_words=num_words, 
                       embed_size=embed_size,
                       batch_size=batch_size,
                       epochs=epochs)
    false_cross_validation = 0
    auc_callback = auc.RocAucEvaluation(validation_data=(X_val_np, y_val_np),
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

    print("学習が完了しました。履歴を保存しています。")
    with open(MODEL_HISTORY_PATH, 'wb') as handle:
        pickle.dump(history.history, handle)


def pred(test_path):
    try:
        print("テストファイルを読み込んでいます")
        test_df = pd.read_csv(test_path, usecols=["comment_text"])
    except FileNotFoundError:
        sys.exit("テストファイルが見つかりません")
    except ValueError:
        sys.exit("テストファイルデータ内のカラム名が正しくありません。入力テキストのカラム名を\"comment_text\"と指定してください。")
    
    seq_maxlen = 100
    num_words = 20000
    embed_size=128
    X_test_np, tokenizer = _get_test_feature(test_df, seq_maxlen=seq_maxlen, num_words=num_words)
    ftext_wmatrix, glv_wmatrix, unique_rate_np = _get_test_wvector_coeff(test_df, 
                                                                         tokenizer, 
                                                                         num_words=num_words,
                                                                         embed_size=embed_size)
    X_test_np = np.concatenate([X_test_np, unique_rate], axis=1)
    X_test_dict = _get_input_dict(X_test_np, seq_maxlen)

    try:
        print("学習済みモデルを読み込んでいます")
        model = load_model(PRETRAINED_MODEL_PATH)
    except:
        sys.exit("学習済みモデルが読み込めません")
    outputs = model.fit(X_test_dict, batch_size=batch_size, verbose=1
    
    print("テストデータの推論が終了しました。結果を保存しています。")
    test_df["is_toxic"] = outputs
    test_df.to_csv("output.csv", index=False)

def pred_text(input_text):
    # TODO:書く
    pass

def _get_train_feature(train_df, seq_maxlen=100, num_words=20000):
    """
    コメント(文字列) -> トークン化された特徴, ラベル
    TODO: カレントディレクトリに学習済みtokenizerが既に存在している場合は，学習をスキップできるようにする
    """
    train_df["comment_text"] = train_df["comment_text"].apply(lambda text: _get_separeted(text))
    train_np = train_df["comment_text"].values
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train_np))
    # 学習済みtokenizerの保存
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tokened_train_list = tokenizer.text_to_sequences(train_np)
    X_train_np = sequence.pad_sequences(train_np, maxlen=seq_maxlen)
    y_train_np = train_np["is_toxic"].values
        
    return X_train_np, y_train_np, tokenizer

def _get_train_wvector_coeff(train_df, tokenizer, num_words=20000, embed_size=128):
    """
    トークン化した各単語に対応する学習済みfastText, GloVeの重みを抽出。 
    また、学習済みfastText, GloVeに存在しない単語をnew_words_listとして抽出し、
    train_df中にどのくらい新語が含まれているか計算する。
    """
    ftext_wmatrix, new_words_list = _get_weighted_matrix(tokenizer, 
                                                         FTEXT_PRETRAINED_PATH, 
                                                         num_words=num_words,
                                                         embed_size=embed_size)
    glv_wmatrix, _  = _get_weighted_matrix(tokenizer, 
                                           GLV_PRETRAINED_PATH, 
                                           num_words=num_words,
                                           embed_size=embed_size)
    train_df["word_list"] = train_df["comment_text"].apply(lambda text: text.split())
    # word_list中にnew_word_listの単語がどのくらい含まれているか数える
    unique_rate_np = train_df["word_lists"].apply(lambda word_list: 
                                                  np.sum([word for word in word_list if word in new_word_list])
                                                  / len(word_list)).astype("float16").values

    return ftext_wmatrix, glv_wmatrix, unique_rate_np

def _get_separeted(text):
    """Natural Language APIを用いて，文章を単語ごとに分かち書きする．"""
    document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
    syntax_response = client.analyze_syntax(document=document)
    separeted_text = " ".join([s.text.content for s in syntax_response.tokens])
    
    return separeted_text

def _get_weighted_matrix(tokenizer, pretrained_path, num_words=20000, embed_size=128):
    """学習済み単語ベクトルを読み込んで、特徴の重みを計算"""
    embed_idx = dict(_get_coefs(*word_and_vector.strip().split()) for word_and_vector in open(pretrained_path))
    all_embs = np.stack(embed_idx.values())
    embed_mean, embed_std =  all_embs.mean(), all_embs.std()
    word_idx = tokenizer.word_idx
    nb_words = min(num_words, len(word_idx))
    embed_matrix = np.random.normal(embed_mean, embed_std, (nb_words, embed_size))
    new_words_list = []
    for word, idx in word_idx.items():
        if idx >= max_features: continue
        embed_vector = embed_idx.get(word)
        if embed_vector is not None: embed_matrix[i] = embed_vector
        else: new_words_list.append(word)

    return embed_matrix, new_words_list

def _get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def _get_input_dict(input_np, seq_maxlen=100):
    """2種類のモデル（fastText, GloVe）で重み付けできるようにdictを作成"""
    input_dict = {
                  'ftext': input_np[:, :seq_maxlen, :],
                  'glv': input_np[:, :seq_maxlen, :]
                  'uniq_rate': input_np[:, seq_maxlen, :]
                 }

    return input_dict

def _get_model(len_train, fasttext_weight, glove_weight, seq_maxlen=100, num_words=20000, 
               embed_size=128, batch_size=5000, epochs=2):
    inp_ftext = Input(shape=(seq_maxlen, ), name='ftext')
    inp_glv = Input(shape=(seq_maxlen, ), name='glv')
    inp_urate = Input(shape=(1), name='uniq_rate')
    embed_ftext = Embedding(num_words, embed_size, weights=[fasttext_weight])(inp_ftext)
    embed_glv = Embedding(num_words, embed_size, weights=[glove_weight])(inp_glv)
    conc_embed = concatenate()([embed_ftext, embed_glv])
    dout = SpatialDropout1D(0.5)(conc_embed)

    lstmed = Bidirectional(CuDNNLSTM(40, return_sequences=True, go_backwards=True))(dout)
    grued = Bidirectional(CuDNNGRU(40, return_sequences=True, go_backwards=True))(lstmed)
    avg_pooled = GlobalAveragePooling1D()(grued)
    max_pooled = GlobalMaxPooling1D()(grued)
    conc_pool_urate = concatenate()([avg_pooled, max_pooled, inp_urate])
    outs = Dense(1, activation="sigmoid")(conc_pool_urate)

    inputs = [inp_ftext, inp_glv]
    model = Model(inputs=inputs, outputs=x)

    # 重み減数の設定
    exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
    steps = init(len_train / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer_adam,
                  metrics=['accuracy'])

    return model

def _get_test_feature(test_df, seq_maxlen=100, num_words=20000):
    test_df["comment_text"] = test_df["comment_text"].apply(lambda text: _get_separeted(text))
    test_np = test_df["comment_text"].values
    try:
        # 学習済みtokenizerのロード
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError:
        sys.exit("学習済みtokenizerが見つかりません。")
    except:
        sys.exit("tokenizerがロードできません。")
    
    tokened_test_list = tokenizer.text_to_sequences(test_np)
    X_test_np = sequence.pad_sequences(test_np, maxlen=seq_maxlen)
        
    return X_test_np, tokenizer

def _get_test_wvector_coeff(test_df, tokenizer, num_words=20000, embed_size=128):
    try:
        with open(FTEXT_WEIGHTED_MATRIX_PATH, 'rb') as handle:
            ftext_wmatrix = pickle.load(handle)
        with open(GLV_WEIGHTED_MATRIX_PATH, 'rb') as handle:
            glv_wmatrix = pickle.load(handle)
        with open(NEW_WORDS_LIST_PATH, 'rb') as handle:
            new_words_list = pickle.load(handle)
    except:
        print("weighted matrix, new words listが一部または全て存在しないので，作成します。")
        ftext_wmatrix, new_words_list = _get_weighted_matrix(tokenizer, 
                                                             FTEXT_PRETRAINED_PATH, 
                                                             num_words=num_words,
                                                             embed_size=embed_size)
        glv_wmatrix, _  = _get_weighted_matrix(tokenizer, 
                                               GLV_PRETRAINED_PATH, 
                                               num_words=num_words,
                                               embed_size=embed_size)
    finally:
        test_df["word_list"] = test_df["comment_text"].apply(lambda text: text.split())
        # word_list中にnew_word_listの単語がどのくらい含まれているか数える
        unique_rate_np = test_df["word_lists"].apply(lambda word_list: 
                                                     np.sum([word for word in word_list if word in new_word_list])
                                                     / len(word_list)).astype("float16").values

        return ftext_wmatrix, glv_wmatrix, unique_rate_np

if __name__ == '__main__':
    if '--train' in sys.argv:
        try:
            train_path = sys.argv[2]
            train(train_path)
        except IndexError:
            sys.exit("学習データのパスが指定されていません。")
    if '--pred' in sys.argv:
        try:
            test_path = sys.argv[2]
            pred(test_path)
        except IndexError:
            sys.exit("テストデータのパスが指定されていません。")
    if '--pred-text' in sys.argv:
        try:
            input_text = sys.argv[2]
            pred_text(input_text)
        except IndexError:
            sys.exit("推論対象の文字列のパスが指定されていません。")       
