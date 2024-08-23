Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1EhloYWpDsVzieEQtFD_OEV0PW5je_oWD#scrollTo=FoqeeJpSMSlH) ulaşabilirsiniz.
 Dataset : turkish-text-data Kaggle-fthbrmnby
# Sentiment_Classification
## Veri Seti Yükleme
```
import pandas as pd
```
```
from google.colab import drive
drive.mount('/content/drive')
```
```
import pandas as pd
import zipfile
import os

# ZIP dosyasının yolu ve dosyaların çıkarılacağı dizin
zip_dosya_yolu = '/content/drive/MyDrive/turkish-text-data/furki.zip'
cikarma_dizini = '/content/drive/MyDrive/turkish-text-data/cikarilan/'

# Dizini oluştur
os.makedirs(cikarma_dizini, exist_ok=True)

# ZIP dosyasını çıkar
with zipfile.ZipFile(zip_dosya_yolu, 'r') as zip_ref:
    zip_ref.extractall(cikarma_dizini)

# Dosya yollarını belirle
pozitif_dosya_yolu = os.path.join(cikarma_dizini, 'furki.pos')
negatif_dosya_yolu = os.path.join(cikarma_dizini, 'furki.neg')

# Pozitif yorumları yükle ve label ekle
pozitif_df = pd.read_csv(pozitif_dosya_yolu, delimiter='\t', header=None, names=['review'])
pozitif_df['label'] = 'positive'

# Negatif yorumları yükle ve label ekle
negatif_df = pd.read_csv(negatif_dosya_yolu, delimiter='\t', header=None, names=['review'])
negatif_df['label'] = 'negative'

# İki veri çerçevesini birleştir
df = pd.concat([pozitif_df, negatif_df], ignore_index=True)

# Sonuçları göster
print(df.head())

data = df
print(data.head())
```
```
data['label'].value_counts()
```
## Veri Seti Ön İşleme
```
import string
import re
import nltk
from nltk.corpus import stopwords

# NLTK veri kümesini indirin
nltk.download('stopwords')

noktalama = string.punctuation
etkisiz = stopwords.words('turkish')
print(noktalama)
print(etkisiz)
```
```
for d in data['review'].head():
  print(d+ '\n-----------------')
  #etkisiz kelimelerin atilmasi
  temp = ''
  for word in d.split():
    if word not in etkisiz and not word.isnumeric():
      temp += word + ' '
  print(temp + '\n****************')
```
```
for d in data['review'].head():
  print(d+ '\n-----------------')

  temp = ''
  for word in d:
    if word not in noktalama:
      temp += word
  print(temp + '\n****************')
  d = temp
```
## Önişlenmiş Veri Seti
```
data.to_csv(r'./cleaned.csv', index = False)
```
```
# Temizlenmiş veriyi yüklüyoruz
import pandas as pd
data = pd.read_csv('cleaned.csv', sep=",", names=['review', 'label'])
print(data.head())
```
```
import nltk
import string

# Stopwords veri kümesinin zaten indirildiğini kontrol edin
try:
    stopwords = nltk.corpus.stopwords.words('turkish')
except LookupError:
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('turkish')

# Noktalama işaretlerini ve etkisiz kelimeleri alın
noktalama = string.punctuation
etkisiz = stopwords

print("Noktalama işaretleri:")
print(noktalama)
print("\nTürkçe etkisiz kelimeler:")
print(etkisiz)
```
## Veri Setini Bölme
```
# Temizlenmiş veriyi train ve test kümelerine ayırıyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['review'].values.astype('U'),
                                                    data['label'].values.astype('U'), test_size=0.1, random_state=42)
print(x_train.shape)
print(x_test.shape)
```
## Sayma Vektörünü Oluşturma
```
# Train kümesindeki cümlelerin sayma vektörlerini çıkarıyoruz
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
print(x_train_counts.shape)
```
## TF*IDF Vektörü Oluşturma
```
# Train kümesindeki cümlelerin TF*IDF vektörelrini sayma vektörlerinden oluşturuyoruz
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print(x_train_tfidf.shape)
```
## Naive Bayes Model Eğitimi
```
# Çok modlu Niave Bayes sınıflandırıcı eğitiyoruz.clf
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train_tfidf, y_train)
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
```
## Model Performansı Ölçme
```
# Sınıflandırıcı ile test seti üzerinde tahminleme yapıyoruz
y_pred = clf.predict(x_test_tfidf)
for review, sentiment in zip(x_test[:5], y_pred[:]):
  print('%r => %s' % (review, sentiment))
```
## Test Sonuçları
```
# Performans sonuçları
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
```


