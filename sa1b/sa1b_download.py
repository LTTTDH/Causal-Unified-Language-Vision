import os
import argparse
import requests
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and extract files.')
parser.add_argument('--processes', type=int, default=16, help='Number of processes to use for downloading and extracting files.')
parser.add_argument('--input_file', type=str, default='sa1b.txt', help='Path to the input file containing file names and URLs.')
parser.add_argument('--raw_dir', type=str, default='/mnt/ssd/lbk-cvpr/dataset/ShareGPT4V/data/sam', help='Directory to store downloaded files.')
parser.add_argument('--images_dir', type=str, default='/mnt/ssd/lbk-cvpr/dataset/ShareGPT4V/data/sam/images', help='Directory to store extracted image files.')
parser.add_argument('--masks_dir', type=str, default='/mnt/ssd/lbk-cvpr/dataset/ShareGPT4V/data/sam/annotations', help='Directory to store extracted JSON mask files.')
parser.add_argument('--skip_existing', action='store_true', help='Skip extraction if the file has already been extracted')
args = parser.parse_args()

# Read the file names and URLs
with open(args.input_file, 'r') as f:
    lines = f.readlines()[1:]

new_lines = []
for line in lines:
    if line.strip().split('\t')[0] in [f'sa_{i:06d}.tar' for i in range(0, 51)]:
        new_lines.append(line)
new_lines.sort(key=lambda x: int(x.strip().split('\t')[0].split('_')[-1].split('.')[0]))

# Create the directories if they do not exist
os.makedirs(args.raw_dir, exist_ok=True)
os.makedirs(args.images_dir, exist_ok=True)
os.makedirs(args.masks_dir, exist_ok=True)

# Download the file
for i in range(1, 50+1):
    response = requests.get(new_lines[i].strip().split('\t')[1], stream=True)
    with open(os.path.join(args.raw_dir, new_lines[i].strip().split('\t')[0]), 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            f.write(chunk)
    print('file download!: ' + str(new_lines[i].strip().split('\t')[0]))