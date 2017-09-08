package com.ssii.day01


import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed._


object RowTest {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val sparkConf=new SparkConf().setMaster("local").setAppName(s"${this.getClass.getSimpleName}")
  val sc=new SparkContext(sparkConf)
  var rdd1=Vectors.dense(1.0,2.0,3.0)
  var rdd2=Vectors.dense(2.0,3.0,4.0)
  def main(args: Array[String]): Unit = {
   blockMatrix()
  }

  def rowMatrix(): Unit ={
    var rdd3=sc.parallelize(Array(rdd1,rdd2))
    //rdd转换成行矩阵
    var rdd4=new RowMatrix(rdd3)
    println(rdd4.numCols(),rdd4.numRows())
    rdd4.rows.foreach(println)
    println("==============")
    var summary=rdd4.computeColumnSummaryStatistics()
    println(summary.count,summary.max,summary.min,summary.mean,summary.normL1,summary.normL2)
  }
  def indexRow(): Unit ={
    var rdd3=IndexedRow(1,rdd1)
    var rdd4=IndexedRow(2,rdd2)
    var rdd5=sc.parallelize(Array(rdd3,rdd4))
    //rdd转换成索引行矩阵
    var rdd6=new IndexedRowMatrix(rdd5)
    rdd6.rows.foreach(println)
  }
  def coordinateMatrix(): Unit ={
    var ent1= new MatrixEntry(0,1,0.5)
    var ent2= new MatrixEntry(2,2,1.8)
    var entries=sc.parallelize(Array(ent1,ent2))
    //创建一个坐标矩阵
    var coordMat=new CoordinateMatrix(entries)
    coordMat.entries.foreach(println)
    //coordMat进行转置
    var transMat=coordMat.transpose()
    transMat.entries.foreach(println)
    //将坐标矩阵转换成一个索引矩阵
    var  indexedRowMatrix=transMat.toIndexedRowMatrix()
    indexedRowMatrix.rows.foreach(println)
  }
  def blockMatrix(): Unit ={
    var ent1=new MatrixEntry(0,0,1)
    var ent2=new MatrixEntry(1,1,1)
    var ent3=new MatrixEntry(2,0,-1)
    var ent4=new MatrixEntry(2,1,2)
    var ent5=new MatrixEntry(2,2,1)
    var ent6=new MatrixEntry(3,0,1)
    var ent7=new MatrixEntry(3,1,1)
    var ent8=new MatrixEntry(3,3,1)
    var entries=sc.parallelize(Array(ent1,ent2,ent3,ent4,ent5,ent6,ent7,ent8))
    //创建一个坐标矩阵
    var coordMat=new CoordinateMatrix(entries)
    //转换成分布式矩阵
    var blockMat=coordMat.toBlockMatrix(2,2).cache()
    //验证举证
    blockMat.validate()
    //查看矩阵分块情况
    println(blockMat.toLocalMatrix())
    //查看其它分块信息
    println(blockMat.numColBlocks,blockMat.numRowBlocks)
    //计算分布式矩阵和转置矩阵的积矩阵
    println(blockMat.transpose.multiply(blockMat).toLocalMatrix())
  }
}
