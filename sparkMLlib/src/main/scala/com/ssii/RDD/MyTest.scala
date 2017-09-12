package com.ssii.RDD

object MyTest {
  def main(args: Array[String]) {

    println(re(1,10))
  }
  def myTests(): Unit ={
    val num=30
    def plusStep(step: Int) ={
      (num:Int) => num + step
    }
    //给step赋值
    val myFunc = plusStep(3)
    //调用myFunc函数
    println(myFunc(3))
    println(plusStep(2))
  }
  //递归求和函数
  var a=Array(1,2,3,4,5,6,7,8,9)
  def num(b:Int): Int = {
    if(b>a.length-1) 0 else num(b+1)+a(b)
  }
  def re(a: Int,b: Int):Int={
    if (a == b) re(a+1,b-1)+a else if( a > b) 0 else re(a+1,b-1)+a+b
  }
}
