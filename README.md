# blog-toxic-comment-classification

## はじめに

一部のブログには、コメント欄に有害なコメントを多数投下してコミュニティを踏み荒す悪質なユーザ（ひょっとしたら業者さん？）が存在します。
対抗策として、NGワードを設定したり、特定ipのコメントを制限したり、Good/Badの投票機能を用意してフィルタリングしたりする方法がありますが、この手のユーザは巧みにその規制を掻い潜ります。
また、一時的に管理者のみがコメントを閲覧できるようにして、コメントを公開するか否かを決める方法がありますが、その管理には多大な負担がかかります。

執筆者は、Kaggleの[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)で議論されていた手法を参考に，上記の問題の解決に役立つツールを開発しています．本リポジトリでは，収集したblogコメントのデータから、有害/無害なコメントを分類するコードを公開します。

（注：収集したコメントデータ(学習，テスト用)は公開致しませんので、各自ご用意ください）

<br>

## 使い方
### Requirements

- NVIDIAのGPUを搭載したPC (cuDNN LSTM, GRUを用いた分類器を使うため)
- GPU版TensorFlow
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs/?hl=ja)

### 使用手順

学習済みモデル付きのため、以下の手順で有害なコメントを分類できます。

1. 本リポジトリをgit cloneします。
2. [ここ](https://drive.google.com/open?id=0ByFQ96A4DgSPUm9wVWRLdm5qbmc)と[ここ](https://www.dropbox.com/s/7digqy9ag3b9xeu/ja.tar.gz?dl=0)から、学習済みの日本語版分散表現モデルをDLして、git cloneしたディレクトリに置き、2種類のvecファイル名を`fast_neo.vec`, `fast_wiki.vec`に変更します。
3. [ここ](https://drive.google.com/open?id=1Bsx3y12Fu-afbScXEoYQzA5WKqlibhWG)から学習済みモデルをダウンロードして，git cloneしたディレクトリに置きます．
4. blog-toxic-comment-classificationディレクトリに移動して、`$ pip install -r requirements.txt`を実行し、必要なライブラリをインストールします。
5. [問題設定](https://github.com/ababa893/blog-toxic-comment-classification#%E5%95%8F%E9%A1%8C%E8%A8%AD%E5%AE%9A)（次項）を参考に、テストデータ(.csv)を用意します。
6. `$ python classification.py --pred <テストデータのパス>`を実行します。
7. 結果がcsvファイルとして、同ディレクトリ内に出力されます。

手順6において、次のように1コメントずつ指定することもできます。

```
$ python classification.py --pred-text "キモいブスばっか全員消えろ"
Toxicity of キモいブスばっか全員消えろ is 0.93445

$ python classification.py --pred-text "バカハゲ間抜けカス"
Toxicity of バカハゲ間抜けカス is 0.97665
```

学習を行いたい場合は、以下を実行します。

```$ python train.py --train <学習データのパス>```

モデルは、同ディレクトリ内に保存されます。

<br>

## 問題設定

次のように、"comment_text"カラムに文字列が入力データとして与えられ、この入力データに対して、"is_toxic"カラムにコメントを
`1:有害, 0:無害`で定義します。 


![データの構成](https://github.com/ababa893/blog-toxic-comment-classification/blob/images/data.png?raw=true)


この形式のデータセットを用いて、未知の"comment_text"入力が与えられたときに"is_toxic"（有害度）を予測する分類器をつくります。


![分類器](https://github.com/ababa893/blog-toxic-comment-classification/blob/images/classifier.png?raw=true)


<br>


## 特徴抽出の手法

### テキストのベクトル化

Kerasの[Tokenizer](https://keras.io/ja/preprocessing/text/)クラスを用いて、単語を数値に変換して文字列をベクトルとして扱います。

日本語の文章は、単語の間にスペースが入っていないので、文字列をベクトルとして扱うには、予め単語同士を分かち書きする必要があります。分かち書きには、[Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs/?hl=ja)を用います。

文字列をTokenizerクラスで単純にベクトル化するだけでも、それなりに良好な性能を発揮しますが、更に汎化性能を上げたいので、学習済み分散表現モデルによって重み付けを行います。

### 学習済み分散表現モデルによる重みづけ

分散表現(単語埋め込み、 Word embedding)とは、単語を200次元等の低次な実数ベクトルで表現する技術です。
代表的な分散表現モデルに[Skip-gram, CBOW](https://arxiv.org/abs/1411.2738), [fasttext](https://github.com/facebookresearch/fastText), [GloVe](https://nlp.stanford.edu/projects/glove/)があります。
 
#### 分散表現で何が嬉しいか

 - 近い意味の単語を、近いベクトルとして表現でき、[単語の散らばり具合を可視化](https://sites.google.com/site/iwanamidatascience/_/rsrc/1468857206744/vol2/word-embedding/words.5k.thumbnail.png?height=600&width=600)できます。
 - ベクトル同士の演算で意味の関係を表現できます。（例：　図書館 - 本 = ホール）

#### 使用する分散表現モデル

[Toxic Comment Classification Challengeの上位者のコメント](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644)によると、複数の分散表現で重み付けした方がより高性能な分類器を構築できるようです。~~そこで、提案する有害コメント分類器では、fastTextとGloVeを両方使用しています。~~
そこで、異なるコーパスで学習させた2種類のfastTextモデルを使用しています。

### コーパス

~~まとめブログを中心に、PV数の多いブログのコメントを収集してコーパスを用意し、分散表現モデルを学習しました。本リポジトリでは、約~MBのコーパスを学習させています。~~

学習済みの日本語版[fastText (NEologd)](https://qiita.com/Hironsan/items/8f7d35f0a36e0f99752c#fasttext)、[fastText (Wikipedia)](https://github.com/Kyubyong/wordvectors#pre-trained-models)を利用します。

<br>

## 分類器

コメントは単語の系列データとして扱われるため、分類器にはBidirectional LSTM, Bidirectional GRUを組み合わせた、ニューラルネットワークモデルを構成します。構成は以下のようになります。

![モデルの構成](https://github.com/ababa893/blog-toxic-comment-classification/blob/images/model_only_ftext.png?raw=true)

ポイントは、第1層と第6層です。第1層では、**"comment_text"をfastText**~~・GloVeモデル~~**を用いて重み付けした2種類の特徴量**を連結させています。第6層では、"comment_text"の1サンプルにおいて、**学習済み分散表現モデルに登録していない新出単語が混入している比率**を計算した特徴量を連結させています。

<br>

## 結果

### データ
某有名まとめブログから，最近の記事に対するコメント約13万を抽出したものを，学習データとして使用しました．
ラベルデータは，執筆者が手作業で有害度の高そうなコメントを直観的に`1`としました．

### 計算環境
- Windows10 64bit
- NVIDIA GeForce GTX1060 6GB

### パラメータ
ハイパーパラメータはチューニングしていません．仮に以下の値を設定しました．
- learning rate 初期値 0.002, 最終値 0.0002
- ベクトル化したコメントの次元数 300
- Embedding層 出力の次元数 300
- バッチサイズ 500
- dropout 0.5
- LSTM, GRU 出力の次元数 40

### 計算結果
学習データを全体の95%，バリデーションデータを5%，epochs=2として学習を行った結果は以下の通りです．

```
Train on 131854 samples, validate on 6940 samples
Epoch 1/2
131854/131854 [==============================] - 97s 732us/step - loss: 0.0109 - acc: 0.6767 - val_loss: 0.1627 - val_acc: 0.9594
6940/6940 [==============================] - 2s 223us/step
¥n ROC-AUC - epoch: 1 - score: 0.987343

Epoch 2/2
131854/131854 [==============================] - 94s 713us/step - loss: 0.0033 - acc: 0.9515 - val_loss: 0.1561 - val_acc: 0.9454
6940/6940 [==============================] - 1s 192us/step
¥n ROC-AUC - epoch: 2 - score: 0.990649
```

### 適当にコメントの有害度を推論

```
Toxicity of うわあああああネトウヨブサヨキモオタうんちして嫌韓厨，嫌煙野郎，ま～ん，ニート，童貞，低学歴！ is  0.9993185

Toxicity of 世の中いろいろな人がいるんだね is  0.1884836

Toxicity of 僕はね、音楽なんてどうでもよくて君のことが好きなんやけど、でも、あの、その、だから楽器を握るんじゃなくて、
君の手を握りたいけど、だけれども、だけれでも、僕はもう、こうやって音楽を奏でて、君に言葉を伝えるその術しか持ってないから、
僕は君の為に、歌う、も、ぼ、僕のために歌いたいんです！ is  0.007785824

Toxicity of どういう場面で使えばいいかよくわかりませんなｗｗｗ元ネタも不明ですぞｗｗｗ is  0.035242625

Toxicity of きっとショボいポエムとか添えてたんだろーなwキモいw is  0.21410456
```


## TODO

- GloVeの学習済み分散表現モデルをつくる
- 他の手法(GBDT，ナイーブベイズ分類器，ロジスティック回帰，NB-SVM etc.)と性能の比較






