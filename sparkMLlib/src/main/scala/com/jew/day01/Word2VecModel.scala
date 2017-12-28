package com.jew.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/27.
  */
object Word2VecModel {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    //创建三个词语序列，每个代表一个文档
    val documentDF = spark.createDataFrame(Seq(
            "Hi I heard about Spark".split(" "),
         "I wish Java could use case classes".split(" "),
         "Logistic regression models are neat".split(" ")
        ).map(Tuple1.apply)).toDF("text")
    documentDF.select("text").show()
    //它是一个Estimator，设置相应的超参数，这里设置特征向量的维度为3
    val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(10).setMinCount(0)
    //读入训练数据，用fit()方法生成一个Word2VecModel
    val model = word2Vec.fit(documentDF)
    //把文档转变成特征向量
    val result =model.transform(documentDF)
    result.select("result").take(3).foreach(println)
  }
}
