# blog-toxic-comment-classification

<借置き中、未完成>

## はじめに

一部のブログには、コメント欄に有害なコメントを多数投下してコミュニティを踏み荒すユーザ（ひょっとしたら業者さん？）が存在します。
対抗策として、NGワードを設定したり、特定ipのコメントを制限したり、Good/Badの投票機能を用意したりする方法がありますが、この手のユーザは巧みにその規制を掻い潜ります。
また、一時的に管理者のみがコメントを閲覧できるようにして、コメントを公開するか否かを決める方法がありますが、管理に多大な負担を強います。

執筆者は、Kaggleの[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)で議論されていた手法を適用することで、この問題を解決できるではないかと考えました。

本リポジトリでは、収集したblogコメントのデータから、有害/無害なコメントを分類するコードを公開します。

（注：収集したコメントデータは公開致しませんので、各自ご用意ください）

<br>

## 使い方
### Requirements

- NVIDIA製GPUを搭載したPC (cuDNN LSTM, GRUを用いた分類器を使うため)
- GPU版TensorFlowが動作する環境 
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs/?hl=ja)

### 使用手順

学習済みモデル付きのため、以下の手順で有害なコメントを分類できます。

1. 本リポジトリをgit cloneします。
2. blog-toxic-comment-classificationディレクトリに移動して、`$ pip install -r requirements.txt`を実行し、必要なライブラリをインストールします。
3. `$ python classification.py --pred <テストデータのパス>`を実行します。
4. 結果がcsvファイルとして、同ディレクトリ内に出力されます。

手順3において、次のように1コメントずつ指定することもできます。

```
$ python classification.py --pred-text "キモいブスばっか全員消えろ"
Toxicity of "キモいブスばっか全員消えろ" is 0.93445

$ python classification.py --pred-text "バカハゲ間抜けカス"
Toxicity of "バカハゲ間抜けカス" is 0.97665
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

Kerasの[Tokenizer](https://keras.io/ja/preprocessing/text/)クラスを用いて，単語を数値ラベルに変換して文字列をベクトルとして扱います．

日本語の文章は、単語の間にスペースが入っていないので、文字列をベクトルとして扱うには、予め単語同士を分かち書きする必要があります。分かち書きには、[Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs/?hl=ja)を用います．

文字列をTokenizerクラスで単純にベクトル化するだけでも，それなりに良好な性能を発揮しますが，更に汎化性能を上げたいので，学習済み分散表現モデルによって重み付けを行います．

### 学習済み分散表現モデルによる重みづけ

分散表現(単語埋め込み、 Word embedding)とは、単語を200次元等の低次な実数ベクトルで表現する技術です。
代表的な分散表現モデルに[Skip-gram, CBOW](https://arxiv.org/abs/1411.2738), [fasttext](https://github.com/facebookresearch/fastText), [GloVe](https://nlp.stanford.edu/projects/glove/)があります。
 
#### 分散表現で何が嬉しいか

 - 近い意味の単語を、近いベクトルとして表現でき、[単語の散らばり具合を可視化](https://sites.google.com/site/iwanamidatascience/_/rsrc/1468857206744/vol2/word-embedding/words.5k.thumbnail.png?height=600&width=600)できます。
 - ベクトル同士の演算で意味の関係を表現できます。（例：　図書館 - 本 = ホール）

#### 使用する分散表現モデル

[Toxic Comment Classification Challengeの上位者のコメント](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644)によると、複数の分散表現モデルで重み付けした方がより高性能な分類器を構築できるようです。そこで、提案する有害コメント分類器では、fastTextとGloVeを両方使用しています。

### コーパス

~~まとめブログを中心に、PV数の多いブログのコメントを収集してコーパスを用意し、分散表現モデルを学習しました。本リポジトリでは、約~MBのコーパスを学習させています。~~
学習済みの日本語版[fastText](https://qiita.com/Hironsan/items/8f7d35f0a36e0f99752c#fasttext)，[GloVe](https://github.com/Kyubyong/wordvectors#pre-trained-models)を利用します．

<br>

## 分類器

コメントは単語の系列データとして扱われるため、分類器にはBidirectional LSTM, Bidirectional GRUを組み合わせた、ニューラルネットワークモデルを構成します。構成は以下のようになります。

![モデルの構成](https://github.com/ababa893/blog-toxic-comment-classification/blob/images/model.png?raw=true)

ポイントは、第1層と第6層です。第1層では、**"comment_text"を学習済みfastText・GloVeモデルを用いて重み付けした2種類の特徴量**を連結させています。第6層では、"comment_text"の1サンプルにおいて、**学習済み分散表現モデルに登録していない新出単語が混入している比率**を計算した特徴量を連結させています。

<br>

## 結果

## TODO
他の分散表現モデルも試す。
時間がないので取り敢えずはfasttext gloveの辞書を使わないかも？





