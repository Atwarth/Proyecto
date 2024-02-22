# import requests

# sample_file = open(r"C:\\Users\\Jorge Eliecer\\Desktop\\2048\\Figure_1.png", 'rb')
# url = "http://192.168.1.106:8000/"
# upload_file = {"Uploaded file": sample_file}
# headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

# r = requests. post(url, headers=headers,files = upload_file)

import requests

sample_file = open(r"C:\\Users\\Jorge Eliecer\\Desktop\\2048\\Figure_1.png", 'rb')
url = "http://192.168.1.106:8080/"
upload_file = {"Uploaded file": sample_file}
headers = {
    "content-type":"image/png",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}

r = requests. post(url, headers=headers, files = upload_file)
#print(r. text)