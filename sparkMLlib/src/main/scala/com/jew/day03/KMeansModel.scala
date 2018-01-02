package com.jew.day03
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
/**
  * Created by Administrator on 2017/12/29.
  */
object KMeansModel {
  case class model_instance (features: Vector)
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    val rawData = spark.sparkContext.textFile("data/mllib/iris.txt")
    val df = rawData.map(line =>
       { model_instance( Vectors.dense(line.split("\t").filter(p => p.matches("\\d*(\\.?)\\d*")).
         map(_.toDouble)) )}).toDF()

  }
}
