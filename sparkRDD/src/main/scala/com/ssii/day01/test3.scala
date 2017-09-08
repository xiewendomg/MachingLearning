package com.ssii.day01

import breeze.linalg.Matrix
import org.apache.spark.ml.linalg.{Matrices, Vectors}


object test3 {
  def main(args: Array[String]): Unit = {
    val dm= Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
    print(dm)
    val sv2 = Vectors.sparse(3, Seq((0, 2.0), (2, 8.0)))
    println(sv2)
    val sm = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))
    println(sm)
  }
}
