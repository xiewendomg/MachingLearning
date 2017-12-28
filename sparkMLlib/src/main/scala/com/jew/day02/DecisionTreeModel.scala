package com.jew.day02

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by xiewendomg on 2017/12/28.
  */
object DecisionTreeModel {
  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    val data = spark.sparkContext.textFile("data/mllib/iris.txt").map(_.split("\t")).map(p =>
      Iris(Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble, p(3).toDouble),p(4).toString())).toDF()
    data.createOrReplaceTempView("iris")
    val df = spark.sql("select * from iris")
    df.map(t => t(1)+":"+t(0)).collect().foreach(println)
    //进一步处理特征和标签，以及数据分组
    //分别获取标签列和特征列，进行索引，并进行了重命名。
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
    println("labelIndexer="+labelIndexer.labels)
    val featureIndexer = new VectorIndexer().setInputCol("features").
      setOutputCol("indexedFeatures").setMaxCategories(4).fit(df)
    //这里我们设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
    val labelConverter = new IndexToString().setInputCol("prediction").
      setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    //接下来，我们把数据集随机分成训练集和测试集，其中训练集占70%。
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    //构建决策树分类模型
    //训练决策树模型,这里我们可以通过setter的方法来设置决策树的参数，
    // 也可以用ParamMap来设置（具体的可以查看spark mllib的官网）。具体的可以设置的参数可以通过explainParams()来获取。
    val dtClassifier = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    //在pipeline中进行设置
     val pipelinedClassifier = new Pipeline().
      setStages(Array(labelIndexer, featureIndexer, dtClassifier, labelConverter))
    //训练决策树模型
    val modelClassifier = pipelinedClassifier.fit(trainingData)
    //进行预测
    val predictionsClassifier = modelClassifier.transform(testData)
    predictionsClassifier.select("predictedLabel", "label", "features").show(20)
  }

}
