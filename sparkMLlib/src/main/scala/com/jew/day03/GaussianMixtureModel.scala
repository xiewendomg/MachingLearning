package com.jew.day03
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.GaussianMixture
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
/**
  * Created by xiewendomg on 2018/1/1.
  */
object GaussianMixtureModel {
  case class model_instance (features: Vector)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "D:\\hadoop-2.7.3")
    val spark = SparkSession.builder().master("local[1]").appName("my App Name").getOrCreate()
    import spark.implicits._
    val rawData=spark.sparkContext.textFile("data/mllib/iris.txt")
    //将数据转换成df
    val df = rawData.map(line =>
       { model_instance( Vectors.dense(line.split("\t").filter(p => p.matches("\\d*(\\.?)\\d*"))
         .map(_.toDouble)) )}).toDF()
    /*
    | 参数 | 含义 |
    | ------- | :----------------: |
    | K | 聚类数目，默认为2 |
    | maxIter | 最大迭代次数，默认为100 |
    | seed | 随机数种子，默认为随机Long值 |
    | Tol | 对数似然函数收敛阈值，默认为0.01 |
      */
    //建立一个简单的GaussianMixture对象，设定其聚类数目为3，其他参数取默认值。
    val gm = new GaussianMixture().setK(3).setPredictionCol("Prediction").setProbabilityCol("Probability")
    val gmm = gm.fit(df)
    //KMeans等硬聚类方法不同的是，除了可以得到对样本的聚簇归属预测外，
    // 还可以得到样本属于各个聚簇的概率（这里我们存在"Probability"列中
    //调用transform()方法处理数据集之后，打印数据集，可以看到每一个样本的预测簇以及其概率分布向量
    val result = gmm.transform(df)
    result.show(150, false)
    for (i <- 0 until gmm.getK) {
       println("Component %d : weight is %f \n mu vector is %s \n sigma matrix is %s" format
         (i, gmm.weights(i), gmm.gaussians(i).mean, gmm.gaussians(i).cov))
       }

  }
}
