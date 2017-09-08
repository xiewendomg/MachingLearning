package com.ssii.MLtype


import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object LabledPoint {

  def main(args: Array[String]): Unit = {
    //创建一个标签为1.0（分类中可视为正样本）的稠密向量标注点
     val pos = LabeledPoint(1.0, Vectors.dense(2.0, 0.0, 8.0))
    //创建一个标签为0.0（分类中可视为负样本）的稀疏向量标注点
     val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(2.0, 8.0)))
  }
}
