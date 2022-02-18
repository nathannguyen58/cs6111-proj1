import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from collections import defaultdict
import constants

nltk.download('punkt')

def processQuery(service, engineID, query, precision, api):
    queryWords = set(query.lower().split(' '))  #tracks the words already in the query
    queryVector = defaultdict(int)      #tracks the current query vector, key: word, value: tf_idf weight


    for word in queryWords:
        queryVector[word] += 1

    res = service.cse().list(q = query, cx = engineID,).execute()

    if (len(res.get('items')) < 10):
        print("Less than 10 documents returned.  The program will terminate now.")
        return

    print("Parameters:")
    print("Client key  = " + api)
    print("Engine key  = " + engineID)
    print("Query       = " + query)
    print("Precision   = " + precision)
    print("Google Search Results:")
    print("======================")

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
            print("")
            

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
            counter += 1

        if (relevanceCounter/(counter-1) >= float(precision)):
            precisionReached = True
            print("======================")
            print("FEEDBACK SUMMARY")
            print("Query " + query)
            print("Precision " + str(relevanceCounter/(counter-1)))
            print("Desired precision reached, done")
            return

        relevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=constants.STOP_WORDS)
        nonrelevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=constants.STOP_WORDS)

        relevantDocumenttfIdf = relevanttfIdfVectorizer.fit_transform(relevantDocumentDataSet)
        nonrelevantDocumenttfIdf = nonrelevanttfIdfVectorizer.fit_transform(nonrelevantDocumentDataSet)

        relevantVector = defaultdict(int)
        nonrelevantVector = defaultdict(int)

        createTFIDFVectorList(queryVector, relevantDocumentVector, relevanttfIdfVectorizer, relevantDocumenttfIdf)
        createTFIDFVectorList(queryVector, nonrelevantDocumentVector, nonrelevanttfIdfVectorizer, nonrelevantDocumenttfIdf)
        
        sortedKeys = rocchio(queryVector, relevantDocumentVector, relevantDocumentDataSet, nonrelevantVector, nonrelevantDocumentDataSet)

        top_two_words = getTopTwoWords(sortedKeys, queryWords)

        sortedKeys = {key: value for key, value in sortedKeys}

        #orderWords(top_two_words, dictOfBigrams)
        originalQuery = query
        query += " " + top_two_words[0] + " " + top_two_words[1]
        orderWords(query, dictOfBigrams, sortedKeys)

        print("======================")
        print("FEEDBACK SUMMARY")
        print("Query " + originalQuery)
        print("Precision " + str (relevanceCounter/(counter-1)))
        print("Still below the desired precision of " + str(precision))
        print("Indexing results ....")
        print("Indexing results ....")
        print("Augmenting by  " + top_two_words[0] + " " + top_two_words[1])
        print("Parameters:")
        print("Client key  = " + api)
        print("Engine key  = " + engineID)
        print("Query       = " + query)
        print("Precision   = " + precision)
        print("Google Search Results:")
        print("======================")

        res = service.cse().list(q = query, cx = engineID,).execute()

def createBigrams(dictOfBigrams, document):
    nltk_tokens = nltk.word_tokenize(document)

    for tuple in list(nltk.bigrams(nltk_tokens)):
        dictOfBigrams[tuple] += 1

def createTFIDFVectorList(queryVector, documentVector, documentVectorizer, documentTfidf):
    for i in range(len(documentTfidf[0].T.todense())):
        documentVector[documentVectorizer.get_feature_names()[i]] = documentTfidf[0].T.todense()[i].tolist()[0][0]

        if documentVectorizer.get_feature_names()[i] not in queryVector:
            queryVector[documentVectorizer.get_feature_names()[i]] = 0

def rocchio(queryVector, relevantVector, relevantDocumentDataSet, nonrelevantVector, nonrelevantDocumentDataSet):
    for word in queryVector.keys():
        queryVector[word] = constants.ALPHA * queryVector[word] + (constants.BETA * relevantVector[word])/len(relevantDocumentDataSet) - (constants.GAMMA * nonrelevantVector[word])/len(nonrelevantDocumentDataSet)
        if queryVector[word] < 0:
            queryVector[word] = 0
        
    return sorted([(word, weight) for word, weight in queryVector.items()], key = lambda x: x[1], reverse = True)

def getTopTwoWords(sortedKeys, queryWords):
    top_two_words = []
    for tuple in sortedKeys:
        if tuple[0] not in queryWords:
            top_two_words.append(tuple[0])
            queryWords.add(tuple[0])
        if len(top_two_words) == 2:
            break
    return top_two_words

def orderWords(query, dictOfBigrams, sortedKeys):
    # if (top_two_words[1], top_two_words[0])  in dictOfBigrams and dictOfBigrams[(top_two_words[1], top_two_words[0])] > dictOfBigrams[(top_two_words[0], top_two_words[1])]:
    #         top_two_words = top_two_words[::-1]
    queryList = query.split(" ")
    queryList.sort(key = lambda x: sortedKeys[x], reverse= True)
    print(queryList)
    for i in range(1, len(queryList)):
        if sortedKeys[queryList[i]] == sortedKeys[queryList[i-1]]:
            if (queryList[i], queryList[i-1]) in dictOfBigrams and dictOfBigrams[(queryList[i], queryList[i-1])] > dictOfBigrams[(queryList[i-1], queryList[i])]:
                temp = queryList[i]
                queryList[i] = queryList[i-1]
                queryList[i-1] = temp
    query = " ".join(queryList)
