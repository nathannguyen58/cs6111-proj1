import pprint
import sys
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from collections import defaultdict
from googleapiclient.discovery import build

nltk.download('punkt')

stopWords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]

def getResults(service, engineID, query, precision):
    queryWords = set(query.lower().split(' '))  #tracks the words already in the query
    queryVector = defaultdict(int)      #tracks the current query vector, key: word, value: tf_idf weight

    alpha = 1
    beta = 0.8
    gamma = 0.1

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

            nltk_tokens = nltk.word_tokenize(currentTitle)

            for tuple in list(nltk.bigrams(nltk_tokens)):
                dictOfBigrams[tuple] += 1
            
            nltk_tokens = nltk.word_tokenize(currentDescription)

            for tuple in list(nltk.bigrams(nltk_tokens)):
                dictOfBigrams[tuple] += 1


            print("")
            counter += 1

        if (relevanceCounter/counter >= float(precision)):
            precisionReached = True
            print('Success!')
            return
        
        print(dictOfBigrams)
        relevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=stopWords)
        nonrelevanttfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=stopWords)

        relevantDocumenttfIdf = relevanttfIdfVectorizer.fit_transform(relevantDocumentDataSet)
        nonrelevantDocumenttfIdf = nonrelevanttfIdfVectorizer.fit_transform(nonrelevantDocumentDataSet)

        relevantVector = defaultdict(int)
        nonrelevantVector = defaultdict(int)

        for i in range(len(relevantDocumenttfIdf[0].T.todense())):
            relevantVector[relevanttfIdfVectorizer.get_feature_names()[i]] = relevantDocumenttfIdf[0].T.todense()[i].tolist()[0][0]

            if relevanttfIdfVectorizer.get_feature_names()[i] not in queryVector:
                queryVector[relevanttfIdfVectorizer.get_feature_names()[i]] = 0
        
        for i in range(len(nonrelevantDocumenttfIdf[0].T.todense())):
            nonrelevantVector[nonrelevanttfIdfVectorizer.get_feature_names()[i]] = nonrelevantDocumenttfIdf[0].T.todense()[i].tolist()[0][0]

            if nonrelevanttfIdfVectorizer.get_feature_names()[i] not in queryVector:
                queryVector[nonrelevanttfIdfVectorizer.get_feature_names()[i]] = 0

        for word in queryVector.keys():
            queryVector[word] = alpha * queryVector[word] + (beta * relevantVector[word])/len(relevantDocumentDataSet) - (gamma * nonrelevantVector[word])/len(nonrelevantDocumentDataSet)
            if queryVector[word] < 0:
                queryVector[word] = 0
        
        sortedKeys = sorted([(word, weight) for word, weight in queryVector.items()], key = lambda x: x[1], reverse = True)

        top_two_words = []


        for tuple in sortedKeys:
            if tuple[0] not in queryWords:
                top_two_words.append(tuple[0])
                queryWords.add(tuple[0])
            if len(top_two_words) == 2:
                break
        
        print(type((top_two_words[1], top_two_words[0])))
        
        if (top_two_words[1], top_two_words[0])  in dictOfBigrams and dictOfBigrams[(top_two_words[1], top_two_words[0])] > dictOfBigrams[(top_two_words[0], top_two_words[1])]:
            top_two_words = top_two_words[::-1]
        print(sortedKeys)
        query += " " + top_two_words[0] + " " + top_two_words[1]
        print(query)
        res = service.cse().list(q = query, cx = engineID,).execute()
    

def main():
    args = sys.argv[1:]
    service = build("customsearch", "v1", developerKey = args[0])
    res = service.cse().list(q = args[3], cx = args[1],).execute()

    getResults(service, args[1], args[3], args[2])

if __name__ == "__main__":
    main()