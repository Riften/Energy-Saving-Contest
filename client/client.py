import requests
url = "http://127.0.0.1:5000/"

files = {'file':('cardboard351.jpg',open('cardboard351.jpg','rb'),'jpg')}
r = requests.post(url,files = files)
print(r)