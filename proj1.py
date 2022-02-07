import pprint
import sys
from collections import defaultdict
from googleapiclient.discovery import build

def getResults(res):
    descriptionList = []
    titleList = []
    urlList = []




    for result in res.get('items'):
        currentTitle = result.get('title')
        currentUrl = result.get('displayLink')
        currentDescription = result.get('snippet')
        

        print(currentTitle)
        print(currentDescription)
        print(currentUrl)

        prompt = input("Is this relevant (Y/N)?")
        
        if (prompt == "Y"):
            descriptionList.append(currentDescription)
            titleList.append(currentTitle)
            urlList.append(currentUrl)

        print("")
    
    commonWords = defaultdict(int)

    for description in descriptionList:   #make sure to remove the words that have already been searched
        words = description.split(" ")
        for word in words:
            commonWords[word] += 1
    
    for title in titleList:   #make sure to remove the words that have already been searched
        words = title.split(" |.|,|;")
        for word in words:
            commonWords[word] += 1

    for word, count in commonWords.items():
        print(word)
        print(count)
    
    print(len(commonWords))
    


    


def main():
    args = sys.argv[1:]
    service = build("customsearch", "v1", developerKey = args[0])
    res = service.cse().list(q = args[3], cx = args[1],).execute()

    getResults(res)
    

    #for item in res:
        #pprint.pprint(item.queries)

    #pprint.pprint(res)

if __name__ == "__main__":
    main()