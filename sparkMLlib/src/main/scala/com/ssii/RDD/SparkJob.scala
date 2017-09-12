
object SparkJob{
  def main(args: Array[String]): Unit = {
    val list=List("abc",12,10.36,'a')
    var aa=list(0)
   var str:String= aa match {
     case _ if("abc".equals(aa)) => "dfdf"
    }
    println(str)
  }
}