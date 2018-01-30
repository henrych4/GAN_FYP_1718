import sys
import os, os.path
from urllib.request import urlretrieve
from multiprocessing.dummy import Pool

f = open(sys.argv[1], 'r', encoding='utf-8')
outdir = sys.argv[1].split('.txt')[0].split('_')[1]
if not os.path.exists(outdir):
    os.makedirs(outdir)
content = f.read().split('\n')

def download_from_url(url):
    try:
        name = url.split('/')[-1]
        urlretrieve(url, './{}/{}'.format(outdir, name))
    except:
        pass

Pool(10).map(download_from_url, content)
totalNumber = len([name for name in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, name))])
print('Finish downloading {} pictures'.format(totalNumber))
