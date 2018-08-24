import os
import numpy as np

class HttpError(Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)

class SystemError(Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)

def download_and_untar(data_url, save_dir):
    if not os.path.exists(save_dir + '.complete.txt'):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for url in data_url:
            file_path = save_dir + os.path.basename(url)

            if os.system('wget -O %s %s' % (file_path, url)):
                raise HttpError("Download error: %s" % (url))
            if os.system('tar xvf %s %s' % (file_path, save_dir)):
                raise SystemError('"tar xvf %s %s" command execution failed.' % (file_path, save_dir))

        np.savetxt(save_dir + '.complete.txt', [len(data_url)])

if __name__ == "__main__":
    download_and_untar(['http://data.cslt.org/thchs30/zip/wav.tgz', 
                        'http://data.cslt.org/thchs30/zip/doc.tgz'], 
                        '/tmp/data_wsj/')
