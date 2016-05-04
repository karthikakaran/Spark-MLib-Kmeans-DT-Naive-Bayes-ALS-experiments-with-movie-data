import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object DecisionTreeClassifier {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeGlassClassifer").setMaster("local")
    val sc = new SparkContext(conf)
    // Load and parse the data file
     val data = sc.textFile("glass.data")
     val parsedData = data.map { line =>
       val parts = line.split(',')
       val features = parts.tail.take(parts.length-2)
       LabeledPoint(parts(10).toDouble, Vectors.dense(features.map(_.toDouble)))
     }
    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)

    // Train a DecisionTree model
    //  Empty categoricalFeaturesInfo indicates all features are continuous
    val numClasses = 8
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    labelAndPreds.foreach(println)
    var rangeLabel = labelAndPreds.values.distinct
    
    //Accuracy of individual classes
    var result:  List[String] = List()
    for( index <- rangeLabel.collect().sorted){
       var classAccuracy = 100 * (1.0 * labelAndPreds.filter(x => x._1 == index.toDouble).count())/(1.0 * labelAndPreds.filter(x => x._2 == index.toDouble).count())
       result = ( index + " : " + Math.min(100.0, classAccuracy) ) :: result
    }
    println("Accuracies for all classes:")
    result.reverse.foreach(println)
   
    //Overall accuracy
    val accuracy = 100 * labelAndPreds.filter(r => r._1 == r._2).count().toDouble / testData.count()
    println("Total Accuracy = " + accuracy)
  }
}
