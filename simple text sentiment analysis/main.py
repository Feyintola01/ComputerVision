from textblob import TextBlob
from newspaper import Article
import nltk

url = 'https://www.nairaland.com/7476492/2023-yari-aggrieved-northern-apc'
# nltk.download('punkt')
article =  Article(url)
article.download()
article.parse()
article.nlp()

texts = article.summary
print(texts)

# with open("text.txt","r") as f:
#   texts = f.read()

blob  = TextBlob(texts)

#return polarity which is in the range of -1.0-1.0
sentiment = blob.sentiment.polarity 
print("The polarity of th article is: ",sentiment)
