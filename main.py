import pprint
import sys
import utils
from googleapiclient.discovery import build

'''
main function: 
takes input from user in the form of <api key>, <engine id>, <precision>, <query> and executes the initial query, 
passes it to the processQuery function in utils.py
'''

def main():    
    args = sys.argv[1:]
    service = build("customsearch", "v1", developerKey = args[0])
    res = service.cse().list(q = args[3], cx = args[1],).execute()

    utils.processQuery(service, args[1], args[3], args[2], args[0])

if __name__ == "__main__":
    main()