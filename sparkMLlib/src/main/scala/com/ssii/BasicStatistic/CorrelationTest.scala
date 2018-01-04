package com.ssii.BasicStatistic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Row, SparkSession}

/**
  * 相关性
  */
object CorrelationTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}")getOrCreate()
    import spark.implicits._
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val data = Seq(
      //稀疏向量和稠密向量可以装换
//      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(1.0, 0.0, 0.0, -2.0),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )

    val df = data.map(Tuple1.apply).toDF("features")
   df.show()
    val Row(coeff1: Matrix) = {
      Correlation.corr(df, "features").head
    }
    println("Pearson correlation matrix:\n" + coeff1.toString)
    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println("Spearman correlation matrix:\n" + coeff2.toString)
  }

}
