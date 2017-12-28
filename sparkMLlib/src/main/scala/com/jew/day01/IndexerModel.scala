package com.jew.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/27.
  */
object IndexerModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  def main(args: Array[String]): Unit = {
   //StringIndexer:索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号
   val df1 = spark.createDataFrame(Seq(
                   (0, "a"),(1, "b"),(2, "c"),
                   (3, "a"),(4, "a"),(5, "c"))).toDF("id", "category")
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")
    //训练
    val model = indexer.fit(df1)
    //该model对DataFrame进行转换操作
    val indexer1=model.transform(df1)
    indexer1.show()
    //df2含有标签转换器不包含的元素“d”
    val df2 = spark.createDataFrame(Seq(
                           (0, "a"),(1, "b"),(2, "c"),(3, "a"),
                           (4, "a"),(5, "d"))).toDF("id", "category")
    //抛出异常
    val indexed = model.transform(df2)
   // indexed.show()
    //setHandleInvalid("skip"),忽略掉那些未出现的标签
   val indexed2 = model.setHandleInvalid("skip").transform(df2)
    indexed2.show()

    // IndexToString的作用是把标签索引的一列重新映射回原有的字符型标签。
    val df = spark.createDataFrame(Seq(
           (0, "a"),(1, "b"),(2, "c"),(3, "a"),
            (4, "a"), (5, "c"))).toDF("id", "category")
    val model3 = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
    val indexed3 = model3.transform(df)
    val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory")
    val converted = converter.transform(indexed3)
    converted.select("id", "originalCategory").show()
  }
}
