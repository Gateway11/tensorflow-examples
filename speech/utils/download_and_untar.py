import os
import numpy as np

def maybe_download_and_untar(data_url, data_dir):
    if not os.path.exists(data_dir + '.complete'):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for url in data_url:
            if url:
                file_path = data_dir + os.path.basename(url)
                if os.system('wget -O %s %s' % (file_path, url)):
                    raise Exception("Download error: %s" % (url))
                print('解压中...')
                if os.system('tar zxf %s -C %s' % (file_path, data_dir)):
                    raise SystemError(
                        '"tar zxf %s %s" command execution failed.' %
                        (file_path, data_dir))
                print('解压完成!')
        np.savetxt(data_dir + '.complete', [len(data_url)])


if __name__ == "__main__":
    download_and_untar(
        ['http://www.openslr.org/resources/18/data_thchs30.tgz'],
        '/tmp/data_wsj/')
