import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object KMeansMovieCluster {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KMeansMovieCluster").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("itemusermat")
    val parsedData = data.map(s => Vectors.dense(s.split(" ").map(_.toDouble))).cache()

    // Cluster the data into 10 classes using KMeans
    val numClusters = 10
    val numIterations = 2
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
   
    val movieRatingCluster = data.map { row =>
      (row.split(" ")(0), clusters.predict(Vectors.dense(row.split(" ").map(_.toDouble))))
    }
  
    val movieAndCluster = movieRatingCluster.map(l => (l._1, l._2))
    
    val movieData = sc.textFile("movies.dat")
    val movieArr = movieData.map(line => (line.split("::")(0), line.split("::")))
    
    val result = movieArr.join(movieAndCluster).distinct().map( t => (t._2._2, t._2._1.toList.mkString(","))).groupByKey()
    result.foreach(c => println("\nCluster: "+c._1+"\n"+c._2.take(5).toList.mkString("\n")))
    
    sc.stop()
  }
}
