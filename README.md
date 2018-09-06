# tensorflow-examples

#### tensorflow build

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package

#### use google colab <https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=t9ALbbpmY9rm>

    #Install CUDA 9
    !curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !apt-get install software-properties-common dirmngr
    !apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    !apt-get update
    !apt-get install cuda
    
    #安装必要库，输入相应代码，并执行
    !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
    !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
    !apt-get update -qq 2>&1 > /dev/null
    !apt-get -y install -qq google-drive-ocamlfuse fuse
    from google.colab import auth
    auth.authenticate_user()
    from oauth2client.client import GoogleCredentials
    creds = GoogleCredentials.get_application_default()
    import getpass
    !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
    vcode = getpass.getpass()
    !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
    
    !mkdir -p drive
    !google-drive-ocamlfuse drive  -o nonempty
    
    #Install cuDNN 7.
    %cd device/Colaboratory
    !tar xzvf cudnn-9.0-linux-x64-v7.2.1.38.tgz
    !cp cuda/lib64/* /usr/local/cuda/lib64/
    !cp cuda/include/cudnn.h /usr/local/cuda/include/
    
    #Install mkdnn
    !apt-get install cmake
    !git clone https://github.com/01org/mkl-dnn.git
    %cd mkl-dnn/scripts
    !./prepare_mkl.sh
    %cd ../
    !mkdir -p build && cd build && cmake .. && make
    !make install
    !echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
    %cd ../..
    
    #run
    !git clone https://github.com/kevinan1/tensorflow-examples.git -b dev
    !pip install python_speech_features
    !pip --no-cache-dir https://github.com/mind/wheels/releases/download/tf1.4.1-gpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
    %cd ./tensorflow-examples/speech
    !python ./run.py

