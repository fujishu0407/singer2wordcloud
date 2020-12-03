import configparser
import functions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

config = configparser.ConfigParser()
config.read('setting.ini')
NEOLOGD = "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
MASK_TELE = "./src/Quote_white.png"
singer_url = 'https://www.uta-net.com/artist/24024/'#須田景凪
#singer_url = 'https://www.uta-net.com/artist/22653/'#ヨルシカ
#singer_url = 'https://www.uta-net.com/artist/12795/'#米津
#singer_url = 'https://www.uta-net.com/artist/26722/'#ずとまよ
#singer_url = 'https://www.uta-net.com/artist/28370/'#YOASOBI
#singer_url = 'https://www.uta-net.com/search/?Aselect=2&Keyword=%E6%9D%B1%E4%BA%AC&Bselect=4&x=32&y=14'#東京



MASKING_MODE = True
COLOR_MODE = False
artist_df = functions.create_dataframe_for_songs(singer_url)
artist_df = functions.add_lyrics_to_dataframe(artist_df)
#preprocess
cd_num_name_dict = {
    "：DUED-123" : "Quote",
    "：WPCL-130" : "porte",
    "：WPCL-129" : "teeter",
    "" : "not album",
}

artist_df.to_csv("./tmp.csv")
artist_df.reset_index(drop=True,inplace=True)

#アルバム単位で歌詞を結合する
lyrics = np.array( [] )
for cd_number in artist_df.CD_Number.unique():
    album = artist_df[artist_df.CD_Number == cd_number].copy()
    lyrics = np.append(lyrics, ' '.join(functions.get_word_list(album.Lyric.tolist(), NEOLOGD)))

#TF-IDFでベクトル化する
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(lyrics)
print(vecs)
words_vectornumber = {}
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    words_vectornumber[v] = k

#各アルバムの各単語のスコアリングをDataFrameにする
vecs_array = vecs.toarray()
albums = []
for vec in vecs_array:
    words_album = []
    vector_album = []
    for i in vec.nonzero()[0]:
        words_album.append(words_vectornumber[i])
        vector_album.append(vec[i])
    albums.append(pd.DataFrame({
        'words' : words_album,
        'vector' : vector_album
    }))

#draw wordcloud all songs words freq
word_list = functions.get_word_list(artist_df.Lyric.tolist(), NEOLOGD)                    
word_freq = pd.Series(word_list).value_counts()
words_df = pd.DataFrame({'noun' : word_freq.index,
             'noun_count' : word_freq.tolist()})
functions.draw_wordcloud(words_df,'noun','noun_count','all songs', MASKING_MODE, COLOR_MODE, MASK_TELE, FONT)
