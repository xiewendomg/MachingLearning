package com.linziyu.day02

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 2017/12/28.
  */
object FeatureSelection {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
  def main(args: Array[String]): Unit = {
    //卡方选择;四个特征维度的数据集，标签有1，0两种
    val df = spark.createDataFrame(Seq(
       (1, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1),
       (2, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0),
       (3, Vectors.dense(1.0, 0.0, 15.0, 0.0), 0)
    )).toDF("id", "features", "label")
    df.show()
    //我们设置只选择和标签关联性最强的一个特征（可以通过setNumTopFeatures(..)方法进行设置
    val selector = new ChiSqSelector().setNumTopFeatures(1).setFeaturesCol("features").
      setLabelCol("label").setOutputCol("selected-feature")
    val selector_model = selector.fit(df)
    val result = selector_model.transform(df)
    //关联性    features&&label 结果关联
    result.show()

  }
}
