package com.jew.day03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
/**
  * Created by xiewendomg on 2018/1/1.
  */
object KMeansModel {
  case class model_instance (features: Vector)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    val rawData=spark.sparkContext.textFile("data/mllib/iris.txt")
    //我们使用了filter算子，过滤掉类标签，正则表达式\\d*(\\.?)\\d*可以用于匹配实数类型的数字，
    // \\d*使用了*限定符，表示匹配0次或多次的数字字符，\\.?使用了?限定符，表示匹配0次或1次的小数点。
    val df = rawData.map( line => {model_instance(
      Vectors.dense(line.split("\t").filter( p => p.matches("\\d*(\\.?)\\d*"))
      .map(_.toDouble))
    )}).toDF()
    //创建KMeans类；训练数据得到模型
    val kmeansmodel = new KMeans().setK(3).setFeaturesCol("features").
      setPredictionCol("prediction").fit(df)
    //
    val results = kmeansmodel.transform(df)
    results.collect().foreach(
      row => {
      println( row(0) + " is predicted as cluster " + row(1))
      })
    //集合内误差平方和
    val t=kmeansmodel.computeCost(df)
    println(t)
  }
}
