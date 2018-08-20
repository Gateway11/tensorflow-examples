【验证码识别】

##### 环境(Mac)
	
	python-3.6
	tensorflow-1.9

##### 依赖

	pip install captcha
	pip install numpy
	pip install opencv-python
	

##### 模型结构
|类型|kernel尺寸/步长(或注释)|输入尺寸|输出尺寸|
|:---:|:---:|:---:|:---:|
|卷积|5 x 5 / 1|60 x 160 x 1|
|池化|2 x 2 / 2|60 x 160 x 32|
|卷积|5 x 5 / 1|30 x 80 x 32|
|池化|2 x 2 / 2|30 x 80 x 64|
|卷积|5 x 5 / 1|15 x 40 x 64|
|池化|2 x 2 / 2|15 x 40 x 64|
|全连接||8 * 20 * 64|
|全连接||8 * 20 * 1024|

##### 运行

	python3 input_data.py
	python3 train.py
	python3 test.py
	
##### 训练结果

	step 3500, training accuracy 0.995469 
