import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from collections import defaultdict

def processQuery(service, engineID, query, precision):
    queryWords = set(query.lower().split(' '))  #tracks the words already in the query
    queryVector = defaultdict(int)      #tracks the current query vector, key: word, value: tf_idf weight


    for word in queryWords:
        queryVector[word] += 1

    res = service.cse().list(q = query, cx = engineID,).execute()

    if (len(res.get('items')) < 10):
        print("Less than 10 documents returned.  The program will terminate now.")
        return

    while (True):
        counter = 1

        relevanceCounter = 0

        dictOfBigrams = defaultdict(int)

        relevantDocumentVector = defaultdict(int)
        nonrelevantDocumentVector = defaultdict(int)
        
        relevantDocumentDataSet = []
        nonrelevantDocumentDataSet = []

        for result in res.get('items'):
            currentTitle = result.get('title')
            currentUrl = result.get('formattedUrl')
            currentDescription = result.get('snippet')
            
            print("Result", counter)
            print("[")
            print("URL:", currentUrl)
            print("Title:", currentTitle)
            print("Summary:", currentDescription)
            print("]")
            

            prompt = input("Relevant (Y/N)?")
            
            currentTitle = currentTitle.lower()
            currentDescription = currentDescription.lower()

            if (prompt == "Y" or prompt == "y"):
                relevanceCounter += 1
                relevantDocumentDataSet.append(currentTitle + " " + currentDescription)

            else:
                nonrelevantDocumentDataSet.append(currentTitle + " " + currentDescription)

            createBigrams(dictOfBigrams, currentTitle)
            createBigrams(dictOfBigrams, currentDescription)


            print("")
            counter += 1

        if (relevanceCounter/counter >= float(precision)):
            precisionReached = True
            print('Success!')
            return
        
        print(dictOfBigrams)
        relevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=constants.STOP_WORDS)
        nonrelevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=constants.STOP_WORDS)

        relevantDocumenttfIdf = relevanttfIdfVectorizer.fit_transform(relevantDocumentDataSet)
        nonrelevantDocumenttfIdf = nonrelevanttfIdfVectorizer.fit_transform(nonrelevantDocumentDataSet)

        relevantVector = defaultdict(int)
        nonrelevantVector = defaultdict(int)

        createTFIDFVectorList(queryVector, relevantDocumentVector, relevanttfIdfVectorizer, relevantDocumenttfIdf)
        createTFIDFVectorList(queryVector, nonrelevantDocumentVector, nonrelevanttfIdfVectorizer, nonrelevantDocumenttfIdf)

        for word in queryVector.keys():
            queryVector[word] = constants.ALPHA * queryVector[word] + (constants.BETA * relevantVector[word])/len(relevantDocumentDataSet) - (constants.GAMMA * nonrelevantVector[word])/len(nonrelevantDocumentDataSet)
            if queryVector[word] < 0:
                queryVector[word] = 0
        
        sortedKeys = rocchio(queryVector, relevantDocumentVector, relevantDocumentDataSet, nonrelevantVector, nonrelevantDocumentDataSet)

        top_two_words = getTopTwoWords(sortedKeys)
        
        print(type((top_two_words[1], top_two_words[0])))

        orderWords(top_two_words, dictOfBigrams)
        print(sortedKeys)
        query += " " + top_two_words[0] + " " + top_two_words[1]
        print(query)
        res = service.cse().list(q = query, cx = engineID,).execute()

def createBigrams(dictOfBigrams, document):
    nltk_tokens = nltk.word_tokenize(document)

    for tuple in list(nltk.bigrams(nltk_tokens)):
        dictOfBigrams[tuple] += 1

def createTFIDFVectorList(queryVector, documentVector, documentVectorizer, documentTfidf):
    for i in range(len(documentTfidf[0].T.todense())):
        relevantVector[documentVectorizer.get_feature_names()[i]] = documentTfidf[0].T.todense()[i].tolist()[0][0]

        if documentVectorizer.get_feature_names()[i] not in queryVector:
            queryVector[documentVectorizer.get_feature_names()[i]] = 0

def rocchio(queryVector, relevantDocumentVector, relevantDocumentDataSet, nonrelevantVector, nonrelevantDocumentDataSet):
    for word in queryVector.keys():
        queryVector[word] = constants.ALPHA * queryVector[word] + (constants.BETA * relevantVector[word])/len(relevantDocumentDataSet) - (constants.GAMMA * nonrelevantVector[word])/len(nonrelevantDocumentDataSet)
        if queryVector[word] < 0:
            queryVector[word] = 0
        
    return sorted([(word, weight) for word, weight in queryVector.items()], key = lambda x: x[1], reverse = True)

def getTopTwoWords(sortedKeys):
    top_two_words = []
    for tuple in sortedKeys:
        if tuple[0] not in queryWords:
            top_two_words.append(tuple[0])
            queryWords.add(tuple[0])
        if len(top_two_words) == 2:
            break
    return top_two_words

def orderWords(top_two_words, dictOfBigrams):
    if (top_two_words[1], top_two_words[0])  in dictOfBigrams and dictOfBigrams[(top_two_words[1], top_two_words[0])] > dictOfBigrams[(top_two_words[0], top_two_words[1])]:
            top_two_words = top_two_words[::-1]
