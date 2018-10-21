import urllib.request
import os
from urllib.parse import urlparse
import shutil

source_dir = 'images'
target_dir = 'data'

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

image_index = 0
for image_index in range(400):
    for filename in os.listdir(source_dir):
        print('Processing ' + filename)
        with open(os.path.join(source_dir, filename), 'r') as file:
            lines = file.readlines()
            if len(lines) > image_index:
                lines = lines[image_index: image_index + 1]
            else:
                continue

            for i, url in enumerate(lines):
                url = url.replace('\n', '')
                print('Retrieving ' + url)
                parsed_url = urlparse(url)
                extension = parsed_url.path.split('.')[-1].split('/')[0]
                if len(extension) == 0:
                    extension = 'jpg'
                target_filename = os.path.join(target_dir, filename.replace('.txt', '-' + str(image_index) + '.' + extension))
                try:
                    urllib.request.urlretrieve(url, target_filename)
                    print('Saved file ' + target_filename)
                except Exception as e:
                    print('Failed to retrieve ' + url + ': ' + str(e))

image_index += 1
