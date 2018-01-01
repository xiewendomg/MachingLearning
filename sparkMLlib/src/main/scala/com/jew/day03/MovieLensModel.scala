package com.jew.day03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2018/1/1.
  */
object MovieLensModel {
//sample_movielens_data.txt
case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  //​ 创建一个Rating类型，即[Int, Int, Float, Long];然后建造一个把数据中每一行转化成Rating类的函数。
  def parseRating(str: String): Rating = {
          val fields = str.split("::")
           assert(fields.size == 3)
           Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
        }
def main(args: Array[String]): Unit = {
  Logger.getLogger("org").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  import spark.implicits._
  val ratings = spark.sparkContext.textFile("data/mllib/sample_movielens_data.txt").map(parseRating).toDF()
  ratings.show()
}
}
