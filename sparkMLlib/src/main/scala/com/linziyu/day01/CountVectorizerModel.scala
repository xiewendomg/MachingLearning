package com.linziyu.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/27.
  */
object CountVectorizerModel {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    //如下的DataFrame，其包含id和words两列，可以看成是一个包含两个文档的迷你语料库。
    val  df=spark.createDataFrame(Seq(
            (0, Array("a", "b", "c")),
            (1, Array("a", "b", "b", "c", "a")))).toDF("id", "words")
    //通过CountVectorizer设定超参数，训练一个CountVectorizerModel，这里设定词汇表的最大量为3，
    // 设定词汇表中的词至少要在2个文档中出现过，以过滤那些偶然出现的词汇。
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").
            setOutputCol("features").setVocabSize(3).setMinDF(2).fit(df)
    println(cvModel.vocabulary)
    //使用这一模型对DataFrame进行变换，可以得到文档的向量化表示
    cvModel.transform(df).show(false)
    //(3,[0,1,2],[1.0,1.0,1.0])  3维度;[0,1,2]索引;[1.0,1.0,1.0]出现次数
    //和其他Transformer不同，CountVectorizerModel可以通过指定一个先验词汇表
    // 来直接生成，如以下例子，直接指定词汇表的成员是“a”，“b”，“c”三个词
    val cvm = new CountVectorizerModel(Array("a", "b", "c")).
             setInputCol("words").
             setOutputCol("features")
    cvm.transform(df).select("features").show()
  }
}
