# tensorflow-examples

#### tensorflow build

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package

#### use google colab <https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=t9ALbbpmY9rm>

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
    
    ## Install CUDA 9
    !curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !apt-get install software-properties-common dirmngr
    !apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    !apt-get update
    !apt-get install cuda
    
    # Install cuDNN 7.
    !tar xzvf device/Colaboratory/cudnn-9.0-linux-x64-v7.2.1.38.tgz
    !cp cuda/lib64/* /usr/local/cuda/lib64/
    !cp cuda/include/cudnn.h /usr/local/cuda/include/
    !apt-get install libcupti-dev
    
    # Install TensorRT
    !dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
    !apt-get update
    !apt-get install tensorrt
    
    !apt-get install cmake
    !git clone https://github.com/01org/mkl-dnn.git
    %cd mkl-dnn/scripts
    !./prepare_mkl.sh
    %cd ../
    !mkdir -p build && cd build && cmake .. && make
    !make install
    %cd ../..
    
    !echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    !echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
    !echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:/usr/local/lib:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc
    !. ~/.bashrc
    
    !git clone https://github.com/kevinan1/tensorflow-examples.git -b dev
    !pip install python_speech_features
    !pip --no-cache-dir https://github.com/mind/wheels/releases/download/tf1.4.1-gpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
    %cd ./tensorflow-examples/speech
    !python ./run.py
