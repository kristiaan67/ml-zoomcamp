#!/usr/bin/env python
# coding: utf-8

import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

data1 = {
    "url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg"
}
data2 = {
    "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
}

results = requests.post(url, json=data1).json()
print(data1['url'], '->', results)

results = requests.post(url, json=data2).json()
print(data2['url'], '->', results)
