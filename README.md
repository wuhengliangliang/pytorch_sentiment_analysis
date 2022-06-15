详细博客介绍：https://blog.csdn.net/qq_41479464/article/details/125286054?spm=1001.2014.3001.5501
# pytorch_sentiment_analysis
基于Pytorch的LSTM实战160万条评论情感分类
数据在百度网盘链接中：

目标：情感分类
数据集 Sentiment140, Twitter上的内容 包含160万条记录，0 ： 负面， 2 ： 中性， 4 ： 正面
但是数据集中没有中性
1、整体流程：
2、导入数据
3、查看数据信息
4、数据预处理：
（统计类别占比（正面和负面）
设置标签和文本
设置表头
样本划分（训练和测试以及验证进行划分数据）
构建词汇表
词汇表大小不一致进行padding）
5、模型构建
6、模型训练

一共160万条评论数据，数据格式如下： 
"0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","TheSpecialOne","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;D" "0","1467810672","Mon Apr 06 22:19:49 PDT 2009","NO_QUERY","scotthamilton","is upset that he can't update his Facebook by texting it... and might cry as a result School today also. Blah!" "0","1467810917","Mon Apr 06 22:19:53 PDT 2009","NO_QUERY","mattycus","@Kenichan I dived many times for the ball. Managed to save 50% The rest go out of bounds" "0","1467811184","Mon Apr 06 22:19:57 PDT 2009","NO_QUERY","ElleCTF","my whole body feels itchy and like its on fire " "0","1467811193","Mon Apr 06 22:19:57 PDT 2009","NO_QUERY","Karoli","@nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there. " "0","1467811372","Mon Apr 06 22:20:00 PDT 2009","NO_QUERY","joy_wolf","@Kwesidei not the whole crew "
![image](https://user-images.githubusercontent.com/39480565/173888360-8c88d280-5016-413a-a731-5ee3774c0d56.png)
