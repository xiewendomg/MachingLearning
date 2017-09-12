package com.ssii.spark2

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

object TFchange {
  val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def stringIndexd(): Unit ={
    //索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。
    val df1 = spark.createDataFrame(Seq(
      (0, "a"), (1, "b"),
      (2, "c"), (3, "a"),
      (4, "a"), (5, "c"))).toDF("id", "category")
    val indexer=new StringIndexer().setInputCol("category")
      .setOutputCol("categoryIndex")
    val model =indexer.fit(df1)
    val indexed1=model.transform(df1)
    //indexed1.show()
    /**
      * DataFrame中有着模型内未曾出现的标签“d”，用已有的模型去转换这一DataFrame会有什么效果？
      */
    val df2 = spark.createDataFrame(Seq(
      (0, "a"), (1, "b"),
      (2, "c"), (3, "a"),
      (4, "a"), (5, "d"))).toDF("id", "category")
    val indexed =model.transform(df2)
    //indexed.show()
    val indexed2=model.setHandleInvalid("skip").transform(df2)
    indexed2.show()
  }
  /**
    *DF-->StringIndexer-->OneHotEncoder
   */
  def oneHotEncoder(): Unit ={
    val df = spark.createDataFrame(Seq(
      (0, "a"), (1, "b"), (2, "c"),
      (3, "a"), (4, "a"), (5, "c"),
      (6, "d"), (7, "d"), (8, "d"),
      (9, "d"), (10, "e"), (11, "e"),
      (12, "e"), (13, "e"), (14, "e")
    )).toDF("id", "category")
    val indexer=new StringIndexer().setInputCol("category")
      .setOutputCol("categoryIndex").fit(df)
    val indexed=indexer.transform(df)
    val encoder=new OneHotEncoder().setInputCol("categoryIndex")
      .setOutputCol("categoryVec").setDropLast(false)
    val encoded=encoder.transform(indexed)
    encoded.show()
  }
  def main(args: Array[String]): Unit = {
    oneHotEncoder()
  }
}
