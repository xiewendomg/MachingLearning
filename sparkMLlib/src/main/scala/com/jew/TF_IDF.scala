package com.jew

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Tokenizer
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

  }
}
