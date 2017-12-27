package com.jew.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 2017/12/27.
  */
object TF_IDF {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    //创建一个简单的DataFram,每一个句子代表一个文档
    val sentenceData = spark.createDataFrame(Seq(
      (0, "I heard about Spark and I love Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
            )).toDF("label", "sentence")
    //得到文档集合后，即可用tokenizer对句子进行分词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.show(false)
    //使用HashingTF的transform()方法把句子哈希成特征向量，这里设置哈希表的桶数为2000
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(2000)
    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.select("rawFeatures").show(false)
    //2000为特征向量维度，240为特征索引值，1.0为出现次数
    //(2000,[240,333,1105,1329,1357,1777],[1.0,1.0,2.0,2.0,1.0,1.0])
    //最后，使用IDF来对单纯的词频特征向量进行修正，使其更能体现不同词汇对文本的区别能力，IDF是一个Estimator，
    // 调用fit()方法并将词频向量传入，即产生一个IDFModel。
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    //IDFModel是一个Transformer，调用它的transform()方法，即可得到每一个单词对应的TF-IDF度量值。
    val rescaledData = idfModel.transform(featurizedData)
    //IDF会减少那些在语料库中出现频率较高的词的权重。
    rescaledData.select("features", "label").take(3).foreach(println)
  }
}
