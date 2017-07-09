import requests
import numpy as np
import pandas
import time
import json

# Setup API key as constant from gitignored file: street_view_api_key
with open('street_view_api_key', 'r') as infile:
    API_KEY = infile.read().replace('\n', '')

# get "<latitude>,longitude" string
def latlng_str(raw_lat, raw_lng):
    lat = str(raw_lat/1e6)
    lng = str(raw_lng/1e6)
    latlng = lat + "," + lng
    return latlng

# get url for location's metadata
def metadata_url(raw_lat, raw_lng):
    latlng = latlng_str(raw_lat, raw_lng)
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?location=" + latlng + "&key=" + API_KEY
    return url

# get url for location's image
def image_url(raw_lat, raw_lng):
    latlng = latlng_str(raw_lat, raw_lng)
    url = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location=" + latlng + "&key=" + API_KEY
    return url

# metadata response tells when image was taken, if it even exists
def metadata_request(url):
    time.sleep(1) # to be polite
    r = requests.get(url)
    return r.json()

# responds with an image
def image_request(url):
    time.sleep(2) # to be polite and prevent from acidentally going significantly over 25,000 requests a day
    r = requests.get(url)
    return r

# parcelid, latitude, and longitude
properties = pandas.read_csv("data/properties_2016.csv", usecols=(0,24,25))

# which images do we already have metadata for
metadata_json = 'data/street_view/metadata.json'
with open(metadata_json, 'r') as infile:
    id2md_orig = json.load(infile)

# percent used to determine how many rows to skip in properties csv
# since only storing metadata for parcels with images
valid_percent = 0.88
next_idx = len(id2md_orig)/valid_percent

# count of the requests that will cost money if doing more than 25,000/day
im_requests = 0
max_per_day = 25000

# dictionary mapping parcelid to street view metadata json
id2metadata = {}
for idx, prop in properties.iterrows():
    if idx < next_idx or str(int(prop['parcelid'])) in id2md_orig:
        print("skip " + str(int(prop['parcelid'])))
        continue
    else:
        if idx > 0:
            print("Requests (now) / Requests (all time) / Row Index: " +
                    str(im_requests) + "/" +
                    str(im_requests+len(id2md_orig)) + "/" +
                    str(idx) +
                    " (" + str((im_requests+len(id2md_orig))/idx) + ")")
        md_url = metadata_url(prop['latitude'], prop['longitude'])
        md_response = metadata_request(md_url)
        if md_response['status'] == 'OK':
            id2metadata[int(prop['parcelid'])] = md_response
            im_url = image_url(prop['latitude'], prop['longitude'])
            im_response = image_request(im_url)

            # save image to im_path
            im_path = 'data/street_view/' + str(int(prop['parcelid'])) + '.jpg'
            with open(im_path, 'wb') as outfile:
                for chunk in im_response.iter_content():
                    outfile.write(chunk)

            # increment image request count
            im_requests += 1
        else:
            print(str(int(prop['parcelid'])) + "," + md_response['status'])

        # update the metadata json file once every hundred downloads
        # to keep from having increasingly long pauses between each
        # image download
        if len(id2metadata) % 100 == 0:
            with open(metadata_json, 'r') as infile:
                id2md_old = json.load(infile)
            id2md_old.update(id2metadata)
            with open(metadata_json, 'w') as outfile:
                json.dump(id2md_old, outfile)
            id2metadata = {}

    # stop requesting images once you hit the daily limit
    if im_requests >= max_per_day:
        break
