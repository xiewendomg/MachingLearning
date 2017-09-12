package com.ssii.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession



object BasicStatistic {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}").getOrCreate()
  val firstRdd=spark.sparkContext.textFile("e:/data.txt")
  def main(args: Array[String]): Unit = {
  summaryStatistic()
  }
def summaryStatistic(): Unit ={
  var rdd1=firstRdd.map(_.split(",")).map(p=>Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble,p(3).toDouble))
  println(rdd1)
  var summary=Statistics.colStats(rdd1)
}

}
