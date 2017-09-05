package com.ssii.RDD

import org.apache.spark.{SparkConf, SparkContext}

object RDDtest {

  def main(args: Array[String]): Unit = {
    val sparkConf=new SparkConf().setMaster("local").setAppName(s"${this.getClass.getSimpleName}")
    val sc=new SparkContext(sparkConf);
    val data=Array( 1 to 9)
    val firstRDD=sc.parallelize(data,3)

  }
}
