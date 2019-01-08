# BeCVer
CV菜鸟立志进步成一名CVer

源代码地址：https://github.com/aymericdamien/TensorFlow-Examples



- 20190107：

  - 加入数据流图的基本操作代码 graph_manage.py
  - 加入变量的基本操作代码 variable_manage.py
  - 加入张量的基本操作代码 tensor_introduce.py
  - 加入TFRecord格式数据生成读取代码：dataset.py

- 20190108：
  - 更新dataset.py笔记：
    - 1、 通过队列进行多线程数据输入：五个基本步骤
      - tf.train.match_filenames_once(正则表达式匹配文件)
      - tf.train.string_input_producer获取输入文件列表
      - tf.train.Coordinatot() 管理过线程的停止：should_stop/request_stop/join
      - tf.train.QueueRunner()队列与线程类、start_queue_runners(sess,coord)
    - 2、使用Dataset框架完成数据输入：三个基本步骤
      - 创建数据集对象：dataset = tf.data.TFRecordDataset()
        - dataset对象的处理函数map/shuffle/batch/repeat
      - 新建可迭代对象：iterator = tf.make_initializable_iterator()
      - 遍历获取数据：   img,label = iterator.get_next()
    - 3、图像处理函数：tf.image