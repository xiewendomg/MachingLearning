package com.ssii.day02

import breeze.linalg.{DenseMatrix, DenseVector, diag}

/**
  * Created by xiewendomg on 2017/11/12.
  */
object test01 {

  def main(args: Array[String]): Unit = {
    /*val m1=DenseMatrix.zeros[Double](2,3)
    println(m1)
    val v1=DenseVector.zeros[Double](3)
    println(v1)
    val v2=DenseVector.ones[Double](3)
    println(v2)
    val v3=DenseVector.fill(3){5.5}
    println(v3)
    val v4=DenseVector.range(1,10,2)
    println(v4)*/
     val m2=DenseMatrix.eye[Double](4)
     println(m2)
     val v6=diag(DenseVector(1.0,2.0,3.0))
     println(v6)
     val m3=DenseMatrix((1.0,2.0),(3.0,4.0))
     println(m3)
     val v8=DenseVector(1,2,3,4)
     println(v8)
     val v9=DenseVector(1,2,3,4).t
     println("v9="+v9(0))
     println(m3.t)
     val v10 = DenseVector.tabulate(3){i=> i*2}
     println(v10)
     val m4=DenseMatrix.tabulate(3,2){(i,j) =>i+j}
    println(m4)
    val v11=new DenseVector(Array(1,2,3,4))
    println(v11)
    val m5=new DenseMatrix(2,3,Array(2,3,4,5,6,7,8))
    println(m5)
    val v12=DenseVector.rand(4)
    println(v12)
    val m6=DenseMatrix.rand(2,3)
    println(m6)
  }
}
