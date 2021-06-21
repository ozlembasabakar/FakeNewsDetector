import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')


dataset = pd.read_csv('news.csv')

corpus = []
for i in range(0, len(dataset)):
  text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
  text = text.lower().split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
  text = ' '.join(text)
  corpus.append(text)


X_train, X_test, y_train, y_test = train_test_split(corpus,dataset['label'], test_size = 0.2, random_state = 0)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)


log_reg = LogisticRegression()
log_reg.fit(tfidf_train,y_train)

pred_lr=log_reg.predict(tfidf_test)

confusion_matrix(y_test, pred_lr)
accuracy_score(y_test, pred_lr)
classification_report(y_test, pred_lr)


knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(tfidf_train, y_train)

pred_knn = knn.predict(tfidf_test)

confusion_matrix(y_test, pred_knn)
accuracy_score(y_test, pred_knn)
classification_report(y_test, pred_knn)


tv = TfidfVectorizer(stop_words='english', max_df=0.7)
x_nb = tv.fit_transform(corpus).toarray()
y_nb = dataset.iloc[:, -1].values

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(x_nb, y_nb, test_size = 0.20, random_state = 0)

nb = GaussianNB()
nb.fit(X_train_nb, y_train_nb)

pred_nb = nb.predict(X_test_nb)

confusion_matrix(y_test_nb, pred_nb)
accuracy_score(y_test_nb, pred_nb)
classification_report(y_test_nb, pred_nb)

    
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(tfidf_train, y_train)

pred_dt = dt.predict(tfidf_test)

confusion_matrix(y_test, pred_dt)
accuracy_score(y_test, pred_dt)
classification_report(y_test, pred_dt)


rf = RandomForestClassifier(n_estimators = 200, random_state=0)
rf.fit(tfidf_train, y_train)

pred_rf = rf.predict(tfidf_test)

confusion_matrix(y_test, pred_rf)
accuracy_score(y_test, pred_rf)
classification_report(y_test, pred_rf)


pac = PassiveAggressiveClassifier()
pac.fit(tfidf_train,y_train)

pred_pac = pac.predict(tfidf_test)

confusion_matrix(y_test, pred_pac)
accuracy_score(y_test, pred_pac)
classification_report(y_test, pred_pac)



le = LabelEncoder()
dataset["label"] = le.fit_transform(dataset["label"]) #0 is Fake, 1 is Real


fake_news = []
real_news = []

test_dataset = dataset.iloc[:,2:] 

for i in range(len(test_dataset)):
    if test_dataset['label'][i] == 0:
        fake_news.append(test_dataset['text'][i])
    else:
        real_news.append(test_dataset['text'][i])


fake_news_words = []
real_news_words = []
for i in range(len(fake_news)):
  text_fake = re.sub('[^a-zA-Z]', ' ', fake_news[i])
  text_fake = text_fake.replace('\n', '')
  text_fake = text_fake.lower().split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  text_fake = [ps.stem(word) for word in text_fake if not word in set(all_stopwords)]
  text_fake = ' '.join(text_fake)
  fake_news_words.append(text_fake)
  
fake_news_words = ''.join(map(str, fake_news_words)) 
  
  
for i in range(len(real_news)):
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')  
  
  text_real = re.sub('[^a-zA-Z]', ' ', real_news[i])
  text_real = text_real.lower().split()
  text_real = [ps.stem(word) for word in text_real if not word in set(all_stopwords)]
  text_real = ' '.join(text_real)
  real_news_words.append(text_real)

real_news_words = ''.join(map(str, real_news_words)) 
  

wordcloud_fake_news = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(fake_news_words)

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud_fake_news) 
plt.axis("off")


wordcloud_real_news = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(real_news_words)

plt.figure(figsize=(40, 30))
plt.imshow(wordcloud_real_news) 
plt.axis("off")

