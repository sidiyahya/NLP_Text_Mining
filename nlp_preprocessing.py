from nltk import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
import nltk
import re
import string

from unidecode import unidecode

stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

##------------PRE-PROCESSING START FROM HERE-------------##
def nlp_prep(text):
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Replace nweline by some space
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    word_tokens = word_tokenize(text)  # n_rows 1971
    stems = ''
    for word in word_tokens:
        stemed_word = stemmer.stem(word)
        if ((stemed_word not in stopwords) and (re.search('[a-zA-Z]', stemed_word)) and stemed_word.isalpha() and len(stemed_word) > 3):
            stems = stems + ' ' + stemed_word

    return stems[1:]  # to remove the first space of the file


# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]


# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


# removes any words composed of less than 2 or more than 21 letters
def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 3 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord


def processCorpus(corpus, language):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)

    for document in corpus:
        index = corpus.index(document)
        corpus[index] = corpus[index].replace(u'\ufffd', '8')  # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(',', '')  # Removes commas
        corpus[index] = corpus[index].rstrip('\n')  # Removes line breaks
        corpus[index] = corpus[index].casefold()  # Makes all letters lowercase

        corpus[index] = re.sub('\W_', ' ', corpus[index])  # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*", " ", corpus[
            index])  # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])  # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])  # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])  # removes URLs with www

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)

        corpus[index] = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus