import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object NaiveBayesClassifier {
  def main(args: Array[String]): Unit = {
     val conf = new SparkConf().setAppName("NaiveBayesClassifier").setMaster("local");
     val sc = new SparkContext(conf)
     val data = sc.textFile("glass.data")
     val parsedData = data.map { line =>
     val parts = line.split(',')
     val features = parts.tail.take(parts.length-2)
       LabeledPoint(parts(10).toDouble, Vectors.dense(features.map(_.toDouble)))
     }
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0)

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    //predictionAndLabel.foreach(println)
    var rangeLabel = predictionAndLabel.values.distinct
    
    //Accuracy of individual classes
    var result:  List[String] = List()
    for( index <- rangeLabel.collect().sorted){
       var classAccuracy = 100 * (1.0 * predictionAndLabel.filter(x => x._1 == index.toDouble).count())/(1.0 * predictionAndLabel.filter(x => x._2 == index.toDouble).count())
       result = ( index + " : " + Math.min(100.0, classAccuracy) ) :: result
    }
    println("Accuracies for all classes:")
    result.reverse.foreach(println)
    
    //Overall accuracy
    var totalAccuracy = 100 * (1.0 * predictionAndLabel.filter(x => x._1 == x._2).count())/test.count()
    println("Total accuracy = %s".format(totalAccuracy))
  }
}
