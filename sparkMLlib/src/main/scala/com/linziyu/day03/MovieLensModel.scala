package com.linziyu.day03

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2018/1/1.
  */
object MovieLensModel {
//sample_movielens_data.txt
case class Rating(userId: Int, movieId: Int, rating: Float)
  //​ 创建一个Rating类型，即[Int, Int, Float, Long];然后建造一个把数据中每一行转化成Rating类的函数。
  def parseRating(str: String): Rating = {
          val fields = str.split("::")
           assert(fields.size == 3)
           Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
        }
def main(args: Array[String]): Unit = {
  Logger.getLogger("org").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  import spark.implicits._
  //读取数据
  val ratings = spark.sparkContext.textFile("data/mllib/sample_movielens_data.txt").map(parseRating).toDF()
  ratings.show()
  //将数据划分成训练集和测试集
  val Array(training,test)=ratings.randomSplit(Array(0.8,0.2))
  //使用ALS来建立推荐模型，这里我们构建了两个模型，一个是显性反馈，一个是隐性反馈
  //显式
  val alsExplicit = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").
    setItemCol("movieId").setRatingCol("rating")
  //隐式
  val alsImplicit = new ALS().setMaxIter(5).setRegParam(0.01).setImplicitPrefs(true).
    setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
  /**
    * numBlocks 是用于并行化计算的用户和商品的分块个数 (默认为10)。
      rank 是模型中隐语义因子的个数（默认为10）。
      maxIter 是迭代的次数（默认为10）。
      regParam 是ALS的正则化参数（默认为1.0）。
      implicitPrefs 决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本（默认是false，即用显性反馈）。
      alpha 是一个针对于隐性反馈 ALS 版本的参数，这个参数决定了偏好行为强度的基准（默认为1.0）。
      nonnegative 决定是否对最小二乘法使用非负的限制（默认为false）。
      可以调整这些参数，不断优化结果，使均方差变小。比如：imaxIter越大，regParam越 小，均方差会越小，推荐结果较优。
    */
    //训练数据集
  val modelExplicit = alsExplicit.fit(training)
  val modelImplicit = alsImplicit.fit(training)
  //模型预测
  //使用训练好的推荐模型对测试集中的用户商品进行预测评分，得到预测评分的数据集
  val predictionsExplicit = modelExplicit.transform(test)
  val predictionsImplicit = modelImplicit.transform(test)
  predictionsExplicit.show()
 }
}
