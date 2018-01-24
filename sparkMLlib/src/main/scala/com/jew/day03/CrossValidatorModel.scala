package com.jew.day03

import com.jew.day02.DecisionTreeModel.Iris
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by xiewendomg on 2018/1/2.
  */
object CrossValidatorModel {
  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    val data = spark.sparkContext.textFile("data/mllib/iris.txt").map(_.split("\t")).map(p =>
      Iris(Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble, p(3).toDouble),p(4).toString())).toDF()
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)
    val lr = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(50)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").
      setLabels(labelIndexer.labels)
    val lrPipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
    val paramGrid = new ParamGridBuilder().
      addGrid(lr.elasticNetParam, Array(0.2,0.8)).
      addGrid(lr.regParam, Array(0.01, 0.1, 0.5)).
      build()
    val cv = new CrossValidator().
      setEstimator(lrPipeline).
      setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")).
      setEstimatorParamMaps(paramGrid).
      setNumFolds(3) // Use 3+ in practice
    val cvModel = cv.fit(trainingData)
    val lrPredictions=cvModel.transform(testData)
    lrPredictions.select("predictedLabel", "label", "features", "probability").
      collect().
      foreach{
        case Row(predictedLabel: String, label:String,features:Vector, prob:Vector) =>
          println(s"($label, $features) --> prob=$prob, predicted Label=$predictedLabel")
      }
    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction")
    val lrAccuracy = evaluator.evaluate(lrPredictions)
    val bestModel= cvModel.bestModel.asInstanceOf[PipelineModel]
    val lrModel = bestModel.stages(2).
      asInstanceOf[LogisticRegressionModel]
    println("Coefficients: " + lrModel.coefficientMatrix + "Intercept: "+lrModel.interceptVector+
      "numClasses: "+lrModel.numClasses+"numFeatures: "+lrModel.numFeatures)
    lrModel.explainParam(lrModel.regParam)
    lrModel.explainParam(lrModel.elasticNetParam)
  }
}
