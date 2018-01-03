package com.linziyu.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/27.
  */
object OneHotEncoderModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  def main(args: Array[String]): Unit = {
    //​独热编码（One-Hot Encoding） 是指把一列类别性特征（或称名词性特征，nominal/categorical features）
    // 映射成一系列的二元连续特征的过程，原有的类别性特征有几种可能取值，这一特征就会被映射成几个二元连续特征，
    // 每一个特征代表一种取值，若该样本表现出该特征，则取1，否则取0。
    val df = spark.createDataFrame(Seq(
      (0, "a"),(1, "b"),(2, "c"),(3, "a"), (4, "a"),(5, "c"),(6, "d"),(7, "d"),
      (8, "d"),(9, "d"),(10, "e"),(11, "e"),(12, "e"),(13, "e"),(14, "e"))).toDF("id", "category")
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
    //原始标签数值化,装换成df
    val indexed = indexer.transform(df)
    //创建OneHotEncoder对象对处理后的DataFrame进行编码，可以看见，编码后的二进制特征呈稀疏向量形式，
    // 与StringIndexer编码的顺序相同，需注意的是最后一个Category（"b"）被编码为全0向量，若希望"b"也占有一个二进制特征，
    // 则可在创建OneHotEncoder时指定setDropLast(false)。
    val encoder = new OneHotEncoder().setInputCol("categoryIndex").setOutputCol("categoryVec").setDropLast(false)
    val encoded = encoder.transform(indexed)
    encoded.show()
  }
}
