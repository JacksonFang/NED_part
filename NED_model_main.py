"""trend news recommendation system"""
import sys
import os
import logging
import pymysql
import base64
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import *
import gensim
from gensim import corpora, models, similarities
from gensim.test.utils import datapath
import re
import numpy as np
from pprint import pprint  # pretty-printer
from flask import Flask, render_template, url_for, redirect, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from datetime import datetime, timedelta

'''input and output description'''

date = list()
title = list()
article = list()
news_link = list()
new_event_bool = list()
dictionary = None
corpus = None
merge = None

caps = list()
top_3_topic = list()
all_topics = list()
kmeans_cluster = list()

old_news = list()
recommendation = list()

tag = list()
intro = list()

date_selected = None
topics = 10
historic = None
current = None


def collect_db_news(d_from, d_to):
    passwd = 'bmeC01OTE='

    date_from = "'" + d_from + "'"
    date_to = "'" + d_to + "'"

    tables = ['thehackernews', 'bleepingcomputer', 'nakedsecurity', 'securelist', 'securityaffairs',
              'welivesecurity', 'hackread']

    global date
    global title
    global article
    global news_link
    global new_event_bool

    date.clear()
    title.clear()
    article.clear()
    news_link.clear()
    new_event_bool.clear()

    p = base64.b64decode(passwd).decode('utf-8')
    conn = pymysql.connect(host='localhost', user='root', passwd=p, db='thehackernews')

    sql = conn.cursor()
    conn.commit()

    for x in tables:
        search = "SELECT * FROM " + x + " WHERE post_date >= " + date_from + " AND post_date <= " + date_to
        print(search)
        sql.execute(search)
        conn.commit()
        for row in sql.fetchall():
            date.append(row[1])
            title.append(row[2])
            article.append(row[3])
            news_link.append(row[4])
            new_event_bool.append(row[5])

    sql.close()
    conn.close()
    print(len(title))


# eliminate same string
# news_link = list(set(news_link))
# article = list(set(article))
# title = list(set(title))


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# turn string into single word
def get_tokens(article):
    punctuation_custom = r"""!"~#$%&'()*+,./:;<=>?@[\]^_`‘’{|}“”"""
    remove_punctuation_map = dict((ord(char), ' ') for char in punctuation_custom)
    no_punctuation = article.translate(remove_punctuation_map)
    tokens = no_punctuation.split()
    # tokens = nltk.word_tokenize(no_punctuation)
    return tokens


# count total words in corpus, with removal of stopwords
def count_term(stopword, article):
    global caps
    temp_caps = []
    tokens = get_tokens(article)
    filtered = []

    for token in tokens:
        # remove numeric word
        pattern = '\d+$'
        match = re.match(pattern, token)
        if match:
            continue
        lower_word = token.lower()
        if lower_word not in stopword:
            filtered.append(lower_word)
            if token[0].isupper():
                temp_caps.append(token)

    caps.append(temp_caps)

    #     lemma = nltk.wordnet.WordNetLemmatizer()
    #     for index, x in enumerate(filtered):
    #         x = lemma.lemmatize(x)
    #         filtered[index] = x

    count = Counter(filtered)
    return count


def output_article_word_list():
    global article
    global title
    global merge
    global caps

    article = list(map(lambda s: s.strip(), article))
    title = list(map(lambda s: s.strip(), title))

    # article = list(map(lambda s: s.replace('\n',''), article))

    for index, x in enumerate(article):
        x = x.replace('\n', '')
        article[index] = x

    countlist = []

    # f = open('/home/fang/downloads/thesis_system_191125/nltk_stopwords', encoding='utf8')

    f = open('/home/fang/downloads/thesis_system_191125/stop_word', encoding='utf8')

    stopword = f.readlines()
    f.close()

    stopword = list(map(lambda s: s.strip(), stopword))

    for x in article:
        countlist.append(count_term(stopword, x))

    # remove duplicate
    for index, x in enumerate(caps):
        caps[index] = list(set(caps[index]))

    # a_list store TF-IDF keyword
    a_list = []
    merge = []
    all_caps = []

    for i, count in enumerate(countlist):
        temp = []
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, count, countlist) for word in count}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:int(len(count) / 2)]:
            temp.append(word)
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 10)))
        a_list.append(temp)

    for x in range(len(a_list)):
        temp_merge = []
        all_caps += caps[x]
        lower = list(map(lambda s: s.lower(), caps[x]))
        #     for i in lower:
        #         if i in a_list[x]:
        #             a_list[x].remove(i)
        #     temp_merge = caps[x] + a_list[x]
        temp_merge = lower + a_list[x]
        temp_merge = list(set(temp_merge))
        #     print(temp_merge)
        merge.append(temp_merge)

    all_caps = list(set(all_caps))

    for x in merge:
        for y in all_caps:
            y_low = y.lower()
            if (y_low in x):
                x.remove(y_low)
                x.append(y)

    pprint(merge)


def build_dictionary_temp():
    global dictionary
    global corpus
    global merge
    dictionary = corpora.Dictionary(merge)
    dictionary.save('/tmp/news_word.dict')  # store the dictionary, for future reference
    print(dictionary)
    dictionary = corpora.Dictionary.load('/tmp/news_word.dict')
    corpus = [dictionary.doc2bow(text) for text in merge]
    corpora.MmCorpus.serialize('/tmp/news_word.mm', corpus)
    # store to disk, for later use

    # print(dictionary.token2id)


def k_means():
    global recommendation
    print("k-means_result")

    top_news = list()

    for index, x in enumerate(all_topics):
        if index in top_3_topic:
            top_news.append(all_topics[index])

    # k-means_with top n topics
    for index, n in enumerate(top_news):
        news_list = []
        ti = []
        for k in n:
            news_list.append(k[1])
            ti.append(k[0])
            print(k[3])
            print(k[0])
            print(k[2])

        vectorizer = TfidfVectorizer(stop_words='english')
        news = vectorizer.fit_transform(news_list)

        true_k = 3
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(news)

        prediction = model.fit_predict(news)
        scores = model.transform(news)
        print(prediction)
        # print(scores)

        for index2, k in enumerate(n):
            # news_list.append(k[1])
            # ti.append(k[0])
            print(k[3])
            print(k[0])
            print(k[2])
            print(prediction[index2])
            print(min(scores[index2]))
            p1 = prediction[index2]
            s1 = (min(scores[index2]))
            l = list(prediction)
            cont = l.count(index2)
            cont2 = 1
            for index3, k2 in enumerate(n):
                p2 = prediction[index3]
                s2 = (min(scores[index3]))
                if p1 == p2:
                    if s1 == s2:
                        news = str(k[3]) + "<br>" + str(k[0]) + "<br>" + str(k[2]) + "<br>" + "<br>"
                    if s1 < s2:
                        news = str(k[3]) + "<br>" + str(k[0]) + "<br>" + str(k[2]) + "<br>" + "<br>"
                        cont2 = cont2 + 1
                    if s1 > s2:
                        cont2 = cont2 + 1

                if cont2 == cont:
                    recommendation.append(news)

    recommendation = list(set(recommendation))
    all_topics.clear()




def recommended_news():
    global recommendation

    print("recommendation")

    for index, x in enumerate(all_topics):
        count = 0
        if index in top_3_topic:
            for i in all_topics[index]:
                print('\n' + 'topic ' + str(index))

                print(i[3])
                print(i[0])
                print(i[2])
                count = count + 1
                news = str(i[3]) + "<br>" + str(i[0]) + "<br>" + str(i[2]) + "<br>" + "<br>"
                # recommendation.append(i[3])
                # recommendation.append(i[0])
                # recommendation.append(i[2])
                recommendation.append(news)

                if count == 3:
                    break
    all_topics.clear()


def flask_web():
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def result():
        global recommendation
        global date_selected
        global topics
        global historic
        global current

        date_selected = request.args.get('date')
        topics = request.args.get('topics')
        historic = request.args.get('historic weeks')
        current = request.args.get('current week')

        if date_selected is not None:
            recommendation.clear()
            d = datetime.strptime(date_selected, "%Y-%m-%d").date()
            history_d = d - timedelta(days=4)
            weekstart = d - timedelta(days=4)
            # collect_db_news(str(history_d), str(weekstart))
            collect_db_news(str(weekstart), str(date_selected))
            output_article_word_list()
            build_dictionary_temp()
            save_longtime_dictionary()
            build_old_lda_model()
            # old_news()

            collect_db_news(str(weekstart), str(date_selected))
            output_article_word_list()
            build_dictionary_temp()
            build_new_lda_model()

            load_lda_model()
            jsd_distance()
            topic_document()
            k_means()

            # recommended_news()
            print(recommendation)

            check = request.args.getlist('check')
            output = recommendation

            print(check)
            print(recommendation)

            with open('selected.txt', 'a') as the_file:
                the_file.write(str(recommendation) + "\n" + str(check) + "\n")

            return render_template("index.html", output=output)

        return render_template("index.html")

    app.run('localhost', 8000)


flask_web()

