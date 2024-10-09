Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1EhloYWpDsVzieEQtFD_OEV0PW5je_oWD#scrollTo=FoqeeJpSMSlH) ulaşabilirsiniz.
 Dataset : turkish-text-data Kaggle-fthbrmnby
# Sentiment_Classification
# Duygu Sınıflandırması (Sentiment Classification)
Bu proje, Türkçe metin verileri üzerinde Naive Bayes algoritması kullanarak duygu sınıflandırması yapmayı amaçlamaktadır. Veri seti, pozitif ve negatif yorumlardan oluşmakta olup, amacımız modelin yeni gelen yorumların duygu durumunu tahmin etmesini sağlamaktır.
## İçindekiler
- Genel Bakış
- Veri
- Veri Ön İşleme
- Modelleme
- Performans
- Kurulum
- Kullanım
- Sonuçlar
- Lisans

## Genel Bakış
Bu projede Türkçe inceleme verileri üzerinde duygu analizi gerçekleştirdim. Verileri temizleyip uygun özniteliklere dönüştürdükten sonra Naive Bayes algoritması ile bir sınıflandırıcı eğitiyoruz. Proje kapsamında TF-IDF vektörleştirme yöntemini kullanarak %90.8 doğruluk elde ettim.

Temel adımlar:

- Veri yükleme ve keşifsel analiz
- Metin ön işleme (noktalama ve stopword'lerin kaldırılması gibi)
- **TF-IDF** kullanarak öznitelik çıkarımı
- **Naive Bayes** sınıflandırıcısı ile model eğitimi
- Model değerlendirmesi ve performans analizi

## Veri
Veri seti, pozitif ve negatif yorumlardan oluşmaktadır, Veri setini arkadaşım düzenlediği için isimlendirmeyi bu şekilde yapmayı tercih ettim.:

- Pozitif Yorumlar: furki.pos
- Negatif Yorumlar: furki.neg
Her dosya satır satır yorumları içerir ve bu yorumların duygu durumunu pozitif veya negatif olarak sınıflandırmayı amaçlıyoruz.

### Örnek Veri 
Önişlemden geçirilmiş veriden birkaç örnek:
| Yorum | Tahmin Edilen Duygu |
| ------ | ------ |
| Bu bir harika film, tavsiye ederim. | positive |
| Kötü bir deneyimdi, zaman kaybıydı. | negative |

## Veri Ön İşleme
Doğal Dil İşleme (NLP) projelerinde, metin ön işleme oldukça kritik bir adımdır. Bu projede verileri temizlemek ve normalize etmek için aşağıdaki işlemleri uyguladık:
1. **Stopword Kaldırma**: NLTK kütüphanesinin Türkçe stopword listesini kullanarak "bir," "ve" gibi yaygın kelimeleri metinden çıkardık.
2. **Noktalama İşaretlerini Kaldırma**: Yorumlardaki noktalama işaretleri temizlendi.
3. **Küçük Harfe Dönüştürme**: Tüm metin küçük harfe dönüştürülerek normalize edildi.

### Önişleme Örneği
Orijinal Yorum:
```bash
Bu filmi izledim ve çok beğendim, kesinlikle öneriyorum!
```
Önişleme Sonrası:
```bash
film izledim beğendim kesinlikle öneriyorum
```
## Modelleme
Projede **Multinomial Naive Bayes** sınıflandırma algoritması kullanıldı. Metin verileri önce **TF-IDF (Term Frequency-Inverse Document Frequency)** özniteliklerine dönüştürülüp ardından sınıflandırıcı eğitildi.

Model eğitimi için temel adımlar:

1. **Sayma Vektörleştirme (Count Vectorization)**: Metin, kelime frekanslarına dayalı olarak öznitelik vektörlerine dönüştürüldü.
2. **TF-IDF Dönüşümü**: Kelimelerin ağırlıkları, TF-IDF kullanılarak hesaplandı.
3. **Model Eğitimi**: TF-IDF öznitelikleri kullanılarak **Multinomial Naive Bayes** sınıflandırıcı eğitildi.

## Model Değerlendirmesi
Eğitilen model test kümesi üzerinde değerlendirilmiş ve doğruluk oranı raporlanmıştır. Aşağıda test kümesi üzerinden bazı örnek tahminler yer almaktadır:

| Yorum | Tahmin Edilen Duygu |
| ------ | ------ |
| Bu film gerçekten çok kötüydü. | negative |
| Harika bir deneyim yaşadım. | positive |

## Performans
Model, test verisi üzerinde %90.8 doğruluk oranı elde etmiştir. Bu sonuç, modelin Türkçe metin verileri üzerinde duygu sınıflandırmasında oldukça başarılı olduğunu göstermektedir.

## Kurulum
Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. Bu depoyu klonlayın:
```bash
git clone https://github.com/kullanici-adi/sentiment-classification.git
cd sentiment-classification
```
2. Gerekli Python kütüphanelerini yükleyin:
```bash
pip install -r requirements.txt
```
3. Veritabanını Google Drive'a yükleyin ve script içerisindeki dosya yollarını güncelleyin.
## Kullanım
1. Verisetini Google Drive'a yükleyin ve dosya yollarını güncelledikten sonra aşağıdaki komutu çalıştırarak analizi başlatın:
```bash
python sentiment_classification.py
```
Bu adımlar şunları yapacaktır:

- Veriyi yükleme
- Metni önişleme
- Model eğitimi
- Test seti üzerinde modelin değerlendirilmesi

## Sonuçlar
Naive Bayes sınıflandırıcı kullanarak eğitilen model, test seti üzerinde %90.8 doğruluk elde etti. Bu sonuç, modelin Türkçe inceleme verileri üzerinde duygu sınıflandırmasını başarılı bir şekilde gerçekleştirdiğini gösteriyor.

Sonuç olarak elde edilen doğruluk çıktısı şu şekildedir:
```bash
Test Accuracy: 90.8%
```



