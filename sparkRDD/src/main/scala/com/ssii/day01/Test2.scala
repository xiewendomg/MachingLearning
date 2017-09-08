package com.ssii.day01

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Test2 {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkConf=new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local")
    val sc=new SparkContext(sparkConf)
    val lines=sc.textFile("E:/data")
    //lines.map(_.split("\t")).map(_(0)).foreach(println)
    val rdd13=lines.filter(_.contains("2016-10-13")).map(_.split("\t")).map(p=>(p(1),p(2))).distinct()
    val rdd14=lines.filter(_.contains("2016-10-14")).map(_.split("\t")).map(p=>(p(1),p(2))).distinct()
    val rdd15=rdd13.union(rdd14)
    rdd13.foreach(println)
    println("=======")
    rdd14.foreach(println)
    println("=======")
    //rdd15.foreach(println)
    val rdd16=rdd15.subtract(rdd13).foreach(println)
    //lines.filter(_.contains("2016-10-13")).map(_.split("\t")).map(p=>(p(1),p(2))).foreach(println)
  }
}
