package com.ssii.day01

import org.apache.spark.sql.SparkSession

object FrameAndSet {
  val spark = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .master("local")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val df = spark.read.json("D:/Documents/Downloads/spark-2.2.0-bin-hadoop2.7/examples/src/main/resources/people.json")

    // Displays the content of the DataFrame to stdout
    df.show()
    df.createOrReplaceTempView("people")

    val sqlDF = spark.sql("SELECT * FROM people")
    sqlDF.show()
  }
  // Note: Case classes in Scala 2.10 can support only up to 22 fields. To work around this limit,
  // you can use custom classes that implement the Product interface
  case class Person(name: String, age: Long)

  // Encoders are created for case classes
  val caseClassDS = Seq(Person("Andy", 32)).toDS()
  caseClassDS.show()
  // +----+---+
  // |name|age|
  // +----+---+
  // |Andy| 32|
  // +----+---+

  // Encoders for most common types are automatically provided by importing spark.implicits._
  val primitiveDS = Seq(1, 2, 3).toDS()
  primitiveDS.map(_ + 1).collect() // Returns: Array(2, 3, 4)

  // DataFrames can be converted to a Dataset by providing a class. Mapping will be done by name
  val path = "D:/Documents/Downloads/spark-2.2.0-bin-hadoop2.7/examples/src/main/resources/people.json"
  val peopleDS = spark.read.json(path).as[Person]
  peopleDS.show()
  // +----+-------+
  // | age|   name|
  // +----+-------+
  // |null|Michael|
  // |  30|   Andy|
  // |  19| Justin|
  // +----+-------+
}
