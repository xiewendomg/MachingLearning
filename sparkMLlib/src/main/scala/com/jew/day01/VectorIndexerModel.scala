package com.jew.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/27.
  */
object VectorIndexerModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  def main(args: Array[String]): Unit = {
  //VectorIndexer 解决向量数据集中的类别性特征转换
    val data = Seq(Vectors.dense(-1.0, 1.0, 1.0),Vectors.dense(-2.0, 3.0, 1.0),Vectors.dense(0.0, 5.0, 1.0))
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show()
    val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(2)
    val indexerModel = indexer.fit(df)
    //通过VectorIndexerModel的categoryMaps成员来获得被转换的特征及其映射，
    // 这里可以看到共有两个特征被转换，分别是0号和2号
    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))
    val indexed = indexerModel.transform(df)
    indexed.show()
  }
}
