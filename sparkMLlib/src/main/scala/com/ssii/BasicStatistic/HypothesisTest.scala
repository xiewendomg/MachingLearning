package com.ssii.BasicStatistic

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.SparkSession

/**
  * 假设检验
  */
object HypothesisTest {
  def main(args: Array[String]): Unit = {
    val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}")getOrCreate()
    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )

    import spark.implicits._
    val df = data.toDF("label", "features")
    val chi = ChiSquareTest.test(df, "features", "label").head
    println("pValues = " + chi.getAs[org.apache.spark.ml.linalg.Vector](0))
    println("degreesOfFreedom = " + chi.getSeq[Int](1).mkString("[", ",", "]"))
    println("statistics = " + chi.getAs[org.apache.spark.ml.linalg.Vector](2))
  }
}
