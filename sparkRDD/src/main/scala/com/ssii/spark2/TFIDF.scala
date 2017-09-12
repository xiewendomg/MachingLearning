package com.ssii.spark2

import org.apache.spark.ml.Model
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

/**
  * 特征抽取
  */
object TFIDF {
  val spark=SparkSession.builder().master("local").appName(s"${this.getClass.getSimpleName}").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import spark.implicits._
  def simpleTest(): Unit ={
    val sentenceData = spark.createDataFrame(Seq(
      (0, "I heard about Spark and I love Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.show()
    //将句子哈希成特征向量
    val hashingTF=new HashingTF().setInputCol("words").setOutputCol("rawFeature").setNumFeatures(2000)
    val featureData=hashingTF.transform(wordsData)
    featureData.select("rawFeature").show(false)
    //使用IDF来对单纯的词频特征向量进行修正，使其更能体现不同词汇对文本的区别能力
    val idf=new IDF().setInputCol("rawFeature").setOutputCol("features")
    val idfModel=idf.fit(featureData)
    val rescaledData=idfModel.transform(featureData)
    rescaledData.select("features","label").take(3).foreach(println)
  }
  def word2Vec(): Unit ={
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
       ).map(Tuple1.apply)).toDF("text")
    val word2Vec=new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
    //生成word2vec模型
    val model=word2Vec.fit(documentDF)
    //模型将文档转变成特征向量
    val result=model.transform(documentDF)
    result.select("result").take(3).foreach(println)
  }
  //CountVectorizer旨在通过计数来将一个文档转换为向量。
  def countVectorize(): Unit ={
    //包含id和words两列，可以看成是一个包含两个文档的迷你语料库。
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")
    /**
      * 通过CountVectorizer设定超参数，训练一个CountVectorizerModel，
      * 这里设定词汇表的最大量为3，设定词汇表中的词至少要在2个文档中出现过，以过滤那些偶然出现的词汇。
      */
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)
      //new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2)
    //cvModel
    cvModel.vocabulary.foreach(print)
    cvModel.transform(df).show(false)

  }

  def main(args: Array[String]): Unit = {
    //Tuple1.apply
    countVectorize()
  }
}
