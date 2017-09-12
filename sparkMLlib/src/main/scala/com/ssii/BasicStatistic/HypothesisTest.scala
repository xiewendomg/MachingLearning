package com.ssii.BasicStatistic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{KernelDensity, Statistics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.random.RandomRDDs._
/**
  * 假设检验
  */
object HypothesisTest {
  //Logger.getLogger("org").setLevel(Level.ERROR)
  val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}")getOrCreate()
  val sc=spark.sparkContext
  sc.setLogLevel("ERROR")
  val v1 = sc.textFile("E:/data.txt").map(_.split(",")).map(p => Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble)).first
  val v2 = sc.textFile("E:/data.txt").map(_.split(",")).map(p => Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble)).take(2).last
  val v3=sc.textFile("E:/data.txt")

  def before(): Unit ={
    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )

    import spark.implicits._
    val df1=v3.toDF()
    val ds=v3.toDS()
    val df = data.toDF("label", "features")
    val chi = ChiSquareTest.test(df, "features", "label").head
    println("pValues = " + chi.getAs[org.apache.spark.ml.linalg.Vector](0))
    println("degreesOfFreedom = " + chi.getSeq[Int](1).mkString("[", ",", "]"))
    println("statistics = " + chi.getAs[org.apache.spark.ml.linalg.Vector](2))
  }

  /**
    * 适合度检验
    * mothod：方法
    * statisticj:检验统计量
    * degrees of freedom: 自由度，表示自由变动的样本观测值的数目
    * pValue:P<0.05为显著，P<0.01为非常显著
    */
  def goodness(): Unit ={
    val goodnessOfFitTestResult = Statistics.chiSqTest(v1)
    println(goodnessOfFitTestResult)
  }

  /**
    * 独立性检验
    */
  def indenpendence(): Unit ={
    var mat=Matrices.dense(2,2,Array(v1(0),v1(1),v2(0),v2(1)))
    var a=Statistics.chiSqTest(mat)
    println(a)
    //v1作为样本，把v2作为期望值，进行卡方检验：
    var c1=Statistics.chiSqTest(v1,v2)
    println(c1)
    //键值对进行独立性检验
    val data=sc.textFile("E:/data.txt")
    val obs=data.map { line =>
            val parts = line.split(',')
            LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)=="Iris-versicolor") 1.toDouble else
              2.toDouble, Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts
            (2).toDouble,parts(3).toDouble))
    }
    val featureTestResults=Statistics.chiSqTest(obs)
    var i=1
    featureTestResults.foreach { result =>
         println(s"Column $i:\n$result")
         i += 1
       }
  }

  /**
    * Kolmogorov-Smirnov检验
    */
  def ks(): Unit ={
    val test = sc.textFile("E:/data.txt").map(_.split(",")).map(p => p(0).toDouble)
    val testResult = Statistics.kolmogorovSmirnovTest(test, "norm", 0, 1)
    println(testResult)
    val myCDF: Double => Double = (p=>p*2)
    val testResult2 = Statistics.kolmogorovSmirnovTest(test, myCDF)
    println(testResult2)
  }

  /**
    * 随机数生成
    */
  def randomDataGeneration(): Unit ={
    val u =normalRDD(sc,10000000L, 10)
    val v = u.map(x => 1.0 + 2.0 * x)
    v.take(10).foreach(println)

  }
  /**
    * 核密度计算 Kerbel density estimation
    */
  def kde(): Unit ={
    val test = sc.textFile("E:/data.txt").map(_.split(",")).map(p => p(0).toDouble)
    val kd = new KernelDensity().setSample(test).setBandwidth(3.0)
    val densities = kd.estimate(Array(-1.0, 2.0, 5.0, 5.8))
    densities.foreach(println)
    //println(densities)
  }

  def main(args: Array[String]): Unit = {
    kde()
  }
}
