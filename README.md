# tensorflow-examples

#### tensorflow build

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package


#### use google colab <https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=t9ALbbpmY9rm>

    !apt-get install cmake
    !git clone https://github.com/01org/mkl-dnn.git
    %cd mkl-dnn/scripts
    !./prepare_mkl.sh
    %cd ../
    !mkdir -p build && cd build && cmake .. && make
    !make install
    !echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
    %cd ../..
    
    !git clone https://github.com/kevinan1/tensorflow-examples.git -b dev
    !pip install python_speech_features
    !pip --no-cache-dir https://github.com/mind/wheels/releases/download/tf1.4.1-gpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
    %cd ./tensorflow-examples/speech
    !python ./run.py

