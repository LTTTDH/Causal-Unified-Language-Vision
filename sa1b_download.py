import os
import tarfile
from multiprocessing import Pool
import argparse
import requests

def download_and_extract(args, skip_existing=False):
    file_name, url, raw_dir, images_dir, masks_dir = args
    
    # Download the file
    print(f'Downloading {file_name} from {url}...')
    response = requests.get(url, stream=True)
    with open(f'{raw_dir}/{file_name}', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and extract files.')
parser.add_argument('--processes', type=int, default=8, help='Number of processes to use for downloading and extracting files.')
parser.add_argument('--input_file', type=str, default='sa1b.txt', help='Path to the input file containing file names and URLs.')
parser.add_argument('--raw_dir', type=str, default='/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sam', help='Directory to store downloaded files.')
parser.add_argument('--images_dir', type=str, default='/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sam/images', help='Directory to store extracted image files.')
parser.add_argument('--masks_dir', type=str, default='/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sam/annotations', help='Directory to store extracted JSON mask files.')
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

# Download and extract the files in parallel
with Pool(processes=args.processes) as pool:
    pool.starmap(download_and_extract, [(new_line.strip().split('\t') + [args.raw_dir, args.images_dir, args.masks_dir], args.skip_existing) for new_line in new_lines])

print('All files downloaded successfully!')    
