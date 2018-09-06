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
    
    # Install CUDA 9
    !curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    !apt-get install software-properties-common dirmngr
    !apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    !apt-get update
    !apt-get install cuda
    !ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.9.2 /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.9.0
    
    # Install cuDNN 7.
    !tar zxf drive/Colaboratory/cudnn-9.0-linux-x64-v7.2.1.38.tgz
    !cp -r cuda/lib64/* /usr/local/cuda/lib64/
    !cp cuda/include/cudnn.h /usr/local/cuda/include/
    !apt-get install libcupti-dev
    
    # Install TensorRT
    !dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
    !apt-get update
    !apt-get install tensorrt
    
    # Install MKL
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
    !echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/extras/CUPTI/lib64' >> ~/.bashrc
    !. ~/.bashrc
    
    !pip install tensorflow-gpu==1.9 python_speech_features
    !git clone https://github.com/kevinan1/tensorflow-examples.git -b dev
    %cd ./tensorflow-examples/speech
    !python ./run.py
