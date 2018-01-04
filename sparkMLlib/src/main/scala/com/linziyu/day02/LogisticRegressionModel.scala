package com.linziyu.day02

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession, functions}

/**
  * Created by Administrator on 2017/12/28.
  */
object LogisticRegressionModel {
  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    ///__________ data/mllib/iris.txt
    val data = spark.sparkContext.textFile("data/mllib/iris.txt").map(_.split("\t")).map(p =>
      Iris(Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble, p(3).toDouble), p(4).toString())
    ).toDF()
    data.show(false)
    //df 转换成表
    data.createOrReplaceTempView("iris")
    //筛选lable不是Iris-setosa的
    val df = spark.sql("select * from iris where label != 'Iris-setosa'")
    df.map(t => t(1)+":"+t(0)).collect().foreach(println)
    //​ 分别获取标签列和特征列，进行索引，并进行了重命名。
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(df)
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
    //设置logistic的参数，这里我们统一用setter的方法来设置，也可以用ParamMap来设置（具体的可以查看spark mllib的官网)，
    // 这里我们设置了循环次数为10次，正则化项为0.3
    val lr = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").
      setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    //println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    //设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").
      setLabels(labelIndexer.labels)
    //构建pipeline，设置stage，然后调用fit()来训练模型。
    val lrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
    val lrPipelineModel = lrPipeline.fit(trainingData)
    val lrPredictions = lrPipelineModel.transform(testData)
    //
    lrPredictions.select("predictedLabel", "label", "features", "probability").collect().foreach
    { case Row(predictedLabel: String, label: String, features: Vector, prob: Vector) =>
      println(s"($label, $features) --> prob=$prob, predicted Label=$predictedLabel")}
     //模型评估
    //创建一个MulticlassClassificationEvaluator实例，用setter方法把预测分类的列名和真实分类的列名进行设置；然后计算预测准确率和错误率。
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
    val lrAccuracy = evaluator.evaluate(lrPredictions)//误差率
    println("Test Error = " + (1.0 - lrAccuracy))//正确率
    //调用lrPipelineModel的stages来获取模型
    val lrModel = lrPipelineModel.stages(2).asInstanceOf[LogisticRegressionModel]
    //评估
    println("Coefficients: " + lrModel.coefficients+"Intercept: "+lrModel.intercept
      +"numClasses: "+lrModel.numClasses+"numFeatures: "+lrModel.numFeatures)
    //二项逻辑回归 摘要
    val trainingSummary:LogisticRegressionTrainingSummary= lrModel.summary
    //获得10次循环中损失函数的变化
    val objectiveHistory:Array[Double] = trainingSummary.objectiveHistory
    //损失函数随着循环是逐渐变小的，损失函数越小，模型就越好；
    objectiveHistory.foreach(loss => println(loss))
    //获取用来评估模型性能的矩阵
    val binarySummary:BinaryLogisticRegressionSummary= trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    //areaUnderROC达到了 0.969551282051282，说明我们的分类器还是不错的；
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")
    val fMeasure = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0)
    //通过最大化fMeasure来选取最合适的阈值
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    println("bestThreshold="+bestThreshold)
    lrModel.setThreshold(bestThreshold)
    //我们还可以用多项逻辑斯蒂回归进行多分类分析，多项逻辑斯蒂回归与二项逻辑斯蒂回归类似，
    //只是在模型设置上把 family 参数设置成 multinomial
    val mlr = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").
      setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
    val mlrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, mlr, labelConverter))
    val mlrPipelineModel = mlrPipeline.fit(trainingData)
    val mlrPredictions = mlrPipelineModel.transform(testData)
    mlrPredictions.select("predictedLabel", "label", "features", "probability").collect().foreach
    { case Row(predictedLabel: String, label: String, features: Vector, prob: Vector) =>
      println(s"($label, $features) --> prob=$prob, predictedLabel=$predictedLabel")}
    val mlrAccuracy = evaluator.evaluate(mlrPredictions)
    println("Test Error = " + (1.0 - mlrAccuracy))
    val mlrModel = mlrPipelineModel.stages(2).asInstanceOf[LogisticRegressionModel]
    println("Multinomial coefficients: " + mlrModel.coefficientMatrix+"Multin omial intercepts: "
      +mlrModel.interceptVector+"numClasses: "+mlrModel.numClasses+ "numFeatures: "+mlrModel.numFeatures)

  }
}
