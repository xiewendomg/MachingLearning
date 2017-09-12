package com.ssii.RDD

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]): Unit = {
    val sparkConf=new SparkConf().setAppName(s"${this.getClass.getSimpleName}").setMaster("local")
    val sc=new SparkContext(sparkConf)
    val data =sc.textFile("e:/data.txt")
    val rdd2=data.map(_.split(","))
    val rddfirst = sc.makeRDD(Seq(("dd",132),("sdfsdfsd",5844)))
    //val rdd3=sc.makeRDD(rdd2)
    val wordCount=data.map(_.split(",")).persist().take(2).foreach(
      p=> (p.foreach(println))
    )
   data.map(_.split(",")).partitions.foreach(println)
  }
  def map: Unit ={

  }
}
