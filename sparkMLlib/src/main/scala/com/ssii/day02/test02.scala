package com.ssii.day02

import breeze.linalg.DenseVector

/**
  * Breeze 元素访问
  * Created by xiewendomg on 2017/11/12.
  */
object test02 {

  def main(args: Array[String]): Unit = {
    val a=DenseVector(1,2,3,4,5,6,7,8,9,10)
    //val b=a(0)
    println(a(1))
    /*println("a(1 to 4)=",a(1 to 4))
    println("a(5 to 0 by -1)=",a(5 to 0 by -1))
    println("a(1 to -1)=",a(1 to -1))
    println("a(-1)=",a(-1))*/
  }
}
