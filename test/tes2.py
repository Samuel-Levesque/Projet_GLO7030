import pymongo
# client = pymongo.MongoClient("mongodb+srv://user1:gY5fVQ1CwDL5SDjJ@cluster0-zhhvs.mongodb.net/test?retryWrites=true")
# db = client.test

uri = 'mongodb://example.com/?ssl=true&ssl_cert_reqs=CERT_NONE'
client = pymongo.MongoClient(uri)