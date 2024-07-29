""" This script allows to see how many flickr images are geo-tagged in a specified area without downloading them"""



import re
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm
import urllib.request
import multiprocessing
from urllib.request import urlopen


def get_whole_lines(pattern, lines):
    return [l for l in lines if re.search(pattern, l)]


def get_matches(pattern, lines):
    lines = get_whole_lines(pattern, lines)
    return [re.search(pattern, l).groups()[0] for l in lines]


def parallelize_function(processes_num, target_function, list_object, *args):
    """For each process take a sublist out of list_object and pass it to target_function."""
    assert type(list_object) == list, f"in parallelize_function() list_object must be a list, but it's a {type(list_object)}"
    jobs = []
    processes_num = min(processes_num, len(list_object))
    sublists = np.array_split(np.asarray(list_object, dtype='object'), processes_num)
    # The first process uses tqdm
    sublists[0] = tqdm(sublists[0], ncols=80)
    for process_num in range(processes_num):
        all_arguments = (process_num, processes_num, sublists[process_num], *args)
        p = multiprocessing.Process(target=target_function,
                                    args=all_arguments)
        jobs.append(p)
        p.start()
    for proc in jobs: proc.join()


###############################################################################


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--min_lat", type=float, required=True)
parser.add_argument("--max_lat", type=float, required=True)
parser.add_argument("--min_lon", type=float, required=True)
parser.add_argument("--max_lon", type=float, required=True)
parser.add_argument("--processes_num", type=int, default=8, help="_")

args2 = parser.parse_args()
print(" ".join(sys.argv))
print(args2)


manager = multiprocessing.Manager()


#### First find all IDs of flickr images within the given area.
print("First find all IDs of flickr images within the given area")

def search_flickr_ids_in_bbox(flickr_ids_, lat_, lon_, side_len_):
    url = ("https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key=181073bd1c1766c272d30a9623f96e57&format=rest" +
           f"&bbox={lon_-side_len_}%2C{lat_-side_len_}%2C{lon_}%2C{lat_}&per_page=250")
    for trial_num in range(10):
        try:
            lines = urlopen(url).read().decode('utf-8').split("\n")
            pages_num = int(get_matches("pages=\"(\d+)\"", lines)[0])
            for page_num in range(1, pages_num+2):
                paged_url = url + f"&page={page_num}"
                lines = urlopen(paged_url).read().decode('utf-8').split("\n")
                for flickr_id in get_matches("id=\"(\d+)\" owner=", lines):
                    flickr_ids_[flickr_id] = None
            break
        except urllib.error.HTTPError as e:
            print(f"lat: {lat_}, lon: {lon_}; Exception: {e}")
        except Exception as e:
            print(f"lat: {lat_}, lon: {lon_}; Exception: {e}")

def download_flickr_ids(process_num, processes_num, all_lats_lons_sublist, flickr_ids, side_len):
    for lat, lon in all_lats_lons_sublist:
        search_flickr_ids_in_bbox(flickr_ids, lat, lon, side_len)

# It is necessary to query small areas (i.e. with short side_len), because flickr's APIs have bugs.
# If we query the whole area instead of smaller sub-areas, we'd get only at most few 1000 images.
side_len = 0.001
decimals_num = 3
# flickr_ids is a dict but it is used as a set. This is because there is no manager.set() class
flickr_ids = manager.dict()

lats = np.arange(args2.min_lat, args2.max_lat, side_len)
lons = np.arange(args2.min_lon, args2.max_lon, side_len)
all_lats_lons = []
for lat in lats:
    for lon in lons:
        all_lats_lons.append((round(lat, decimals_num), round(lon, decimals_num)))

random.shuffle(all_lats_lons)  # Shuffle because some areas are dense with images.

parallelize_function(args2.processes_num, download_flickr_ids, all_lats_lons, flickr_ids, side_len)

flickr_ids = sorted(list(flickr_ids.keys()))
print(f"I found {len(flickr_ids)} IDs (aka photos) in this area")