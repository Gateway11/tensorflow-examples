import os
import numpy as np


class HttpError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def download_and_untar(data_url, save_dir):
    if not os.path.exists(save_dir + '.complete.txt'):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('下载中...')
        for url in data_url:
            if url:
                file_path = save_dir + os.path.basename(url)
    
                if os.system('wget -O %s %s' % (file_path, url)):
                    raise HttpError("Download error: %s" % (url))
                print('解压中...')
                if os.system('tar zxf %s -C %s' % (file_path, save_dir)):
                    raise SystemError(
                        '"tar zxf %s %s" command execution failed.' %
                        (file_path, save_dir))
    
        np.savetxt(save_dir + '.complete.txt', [len(data_url)])


if __name__ == "__main__":
    download_and_untar(
        ['http://www.openslr.org/resources/18/data_thchs30.tgz'],
        '/tmp/data_wsj/')
