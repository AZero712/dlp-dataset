import os
import argparse
import uuid
import json
import xml.etree.ElementTree as ET 

import pandas as pd 
import numpy as np

def gen_token():
    # add hash unique to scene
    return uuid.uuid4().hex

parser = argparse.ArgumentParser()
parser.add_argument('file', help='XML file to convert')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
args = parser.parse_args()

tree = ET.parse(args.file)
root = tree.getroot()

if args.verbose:
    print('finished loading XML')

trajectory_df = pd.DataFrame()
filename_stem = os.path.splitext(args.file)[0]

for f in range(len(root)):

    frame = root[f]
    if len(frame) < 2:
        continue

    trajectories = pd.DataFrame()
    trajectories = trajectories.append([{k:t.attrib[k] for k in list(t.attrib.keys())[:11]} for t in frame[:-1]])
    trajectories.insert(0, 'timestamp', frame.attrib['timestamp'])
    trajectories.insert(0, 'frame_id', frame.attrib['id'])
    trajectory_df = trajectory_df.append(trajectories)

    if f % 100 == 0:
        trajectory_df.to_csv(filename_stem + '.csv', mode=('a' if f else 'w'), encoding='utf-8', index=False, header=(not f))
        trajectory_df = pd.DataFrame()

        if args.verbose:
            print('finished frame', f)

trajectory_df.to_csv(filename_stem + '.csv', mode='a', encoding='utf-8', index=False, header=False)