package com.ssii.MLtype

import org.apache.spark.ml.linalg.Vectors

object LocalVector {

  def main(args: Array[String]): Unit = {
    //稠密向量
    val vector1=Vectors.dense(2.0,3.0,4.0)
    //稀疏变量-方法一
    val vector2=Vectors.sparse(3,Array(0,2),Array(2.0,8.0))
    //稀疏变量-方法二
    val vector=Vectors.sparse(3,Seq((0,2.0),(2,8.0)))
    //方法一、二相同
  }
}
