import requests

import hashlib

import time
import datetime

import xmltodict
import json

API_KEY = 'Rd2qJ1LI'
SHARED_SECRET = '1Ehe3mfz'

def get_json_response(response):
    xml_text = response.content
    data_dict = xmltodict.parse(xml_text)
    return data_dict

def get_time_str():
    present = datetime.datetime.now()
    unix_ts = time.mktime(present.timetuple())
    ret = str(int(unix_ts))
    return ret

def get_sha1(s):
    encoded = s.encode('ascii')
    return hashlib.sha1(encoded)

def by_tag():
    url = 'https://www.slideshare.net/api/2/get_slideshows_by_tag'
    ts = get_time_str()
    values = {
            'api_key': API_KEY,
            'ts': ts,
            'hash': get_sha1(SHARED_SECRET + ts).hexdigest(),
            'tag': ['content', 'gary']
        }
    response = requests.get(url, params=values)
    print(response.text)

def by_user():
    url = 'https://www.slideshare.net/api/2/get_slideshows_by_user'
    ts = get_time_str()
    values = {
            'api_key': API_KEY,
            'ts': ts,
            'hash': get_sha1(SHARED_SECRET + ts).hexdigest(),
            #'username_for': 'accenture'
            'username_for': 'elliehood',
        }
    response = requests.get(url, params=values)
    json_data = get_json_response(response)

    print(json_data)

    for slideshow in json_data["User"]["Slideshow"]:
        if "Format" in slideshow:
            print(slideshow['Format'])
            if ("DownloadUrl" in slideshow):
                print(slideshow["DownloadUrl"])


def duckduckgo():
    url = "https://duckduckgo.com/?q=filetype:pptx"
    response = requests.get(url)
    print(response.text)

def main():
    duckduckgo()
    
    
if __name__ == "__main__":
    main()