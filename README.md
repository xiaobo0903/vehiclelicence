# vehiclelicence
1、设置python环境：

   apt -y install python-pip

   pip install flask

   pip install opencv-python

   apt-get install python-numpy  

   apt-get install python-scipy  

   apt-get install python-matplotlib  

   apt-get install python-skimage
   
2、进入安装目录，下载程序:

   git clone https://github.com/xiaobo0903/vehiclelicence.git

3、运行程序：

   python licenceRecognize.py
   
4、访问方式：
    
   http://ip:port/vehicle/licence/recognize?imgurl=urlencode(img-url)

   http://ip:port/vehicle/licence/recognize1?imgurl=urlencode(img-url)
