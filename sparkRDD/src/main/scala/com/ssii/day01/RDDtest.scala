package com.ssii.day01

import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.{SparkConf, SparkContext}

object RDDtest {
  def main(args: Array[String]): Unit = {
    val sparkConf=new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local")
    val sc=new SparkContext(sparkConf)
    val lines=sc.textFile("E:/data")
    //z总行数
    //val number=lines.mapPartitions(_)
    val totalLength = lines.map(_.size).count()
    //用户数量
    var userNum= lines.map(_.split("\t")).map(_(1)).distinct().count()
    //天数哪几天
    var dataNum=lines.map(_.split("\t")).map(_(0)).distinct().foreach(println)
    println("总行数=",totalLength)
    println("用户数量=",userNum)
  }


}
