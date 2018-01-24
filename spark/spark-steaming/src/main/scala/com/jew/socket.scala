package com.jew

import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Created by Administrator on 2018/1/24.
  */
object socket {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir","D:\\hadoop-2.7.3")
    val conf=new SparkConf().setMaster("local[2]").setAppName("socket")
    val ssc= new StreamingContext(conf,Seconds(1))
    val lines =ssc.socketTextStream("spark",9999)
    val word =lines.flatMap(_.split(" "))
    val pairs =lines.map(s=>(s,1))
    val wordCount=pairs.reduceByKey(_+_)
    wordCount.print()
    //启动计算
    ssc.start()
    //等待计算终止
    ssc.awaitTermination()
  }
}
