import breeze.linalg._
import org.apache.spark.HashPartitioner
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object als {
	def main(args: Array[String]){
	  var inputId = ""
	  var option = ""
	  var latMovieOrUser = ""
	  var latUser = ""
	  var userInputId = ""
	  var latMovie = ""
	  var movieInputId = ""
	  
	  if(args.length < 3){
	     println("Usage als <'1/2'>(Latent Vector/Prediction) <'user_id'> <userIdInput> <'movie_id'> <movieIdInput>")
	     exit(2)
	  }
	  else if(args(0).equals("1")){
	    if(args.length < 3){
	      println("Usage als <'1'> <'movie_id/user_id'> <inputId>")
	      exit(2)
	    } else {
	      option = args(0)
	      latMovieOrUser = args(1)
	      inputId = args(2)
	    }
	  }
	  else if(args(0).equals("2")){
	    if(args.length < 5){
	      println("Usage als <'2'> <'user_id'> <userIdInput> <'movie_id'> <movieIdInput>")
	      exit(2)
	    } else {
	      option = args(0)
	      latUser = args(1)
	      userInputId = args(2)
	      latMovie = args(3)
	      movieInputId = args(4)
	    }
	  }
	  
		val conf = new SparkConf().setAppName("als").setMaster("local")
		val sc = new SparkContext(conf)
		//loads ratings from file
		val ratings = sc.textFile("ratings.dat").map(l => (l.split("::")(0),l.split("::")(1),l.split("::")(2))) 

		// counts unique movies
		val itemCount = ratings.map(x=>x._2).distinct.count 

		// counts unique user
		val userCount = ratings.map(x=>x._1).distinct.count 

		// get distinct movies
		val items = ratings.map(x=>x._2).distinct   

		// get distinct user
		val users = ratings.map(x=>x._1).distinct  

		// latent factor
		val k= 5  

		//create item latent vectors
		val itemMatrix = items.map(x=> (x,DenseVector.zeros[Double](k)))   
		//Initialize the values to 0.5
		// generated a latent vector for each item using movie id as key Array((movie_id,densevector)) e.g (2,DenseVector(0.5, 0.5, 0.5, 0.5, 0.5)
		var myitemMatrix = itemMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist  

		//create user latent vectors
		val userMatrix = users.map(x=> (x,DenseVector.zeros[Double](k)))
		//Initialize the values to 0.5
		// generate latent vector for each user using user id as key Array((userid,densevector)) e.g (2,DenseVector(0.5, 0.5, 0.5, 0.5, 0.5)
		var myuserMatrix = userMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist 

		// group rating by items. Elements of type org.apache.spark.rdd.RDD[(String, (String, String))] (itemid,(userid,rating)) e.g  (1,(2,3))
		val ratingByItem = sc.broadcast(ratings.map(x => (x._2,(x._1,x._3)))) 

		// group rating by user.  Elements of type org.apache.spark.rdd.RDD[(String, (String, String))] (userid,(item,rating)) e.g  (1,(3,5)) 
		val ratingByUser = sc.broadcast(ratings.map(x => (x._1,(x._2,x._3)))) 

		var i =0
		for( i <- 1 to 10){
			//=============This code will update the myuserMatrix which contains the latent vectors for each user. 

			// joining the movie latent vector with movie ratings using movieid as key. Step 1 from Sec 14.3 which results in
			//ratItemVec: is an collect of elements of type  org.apache.spark.rdd.RDD[(String, (breeze.linalg.DenseVector[Double], (String, String)))] which means [(movieid, (item latent vector, (user_id, rating)))] e.g  Array((2,(DenseVector(0.5, 0.5, 0.5, 0.5, 0.5),(1,4)))) 
			val ratItemVec = myitemMatrix.join(ratingByItem.value)

					// regularization factor which is lambda.
					val regfactor = 1.0 
					val regMatrix = DenseMatrix.zeros[Double](k,k)  //generate an diagonal matrix with dimension k by k
					//filling in the diagonal values for the reqularization matrix.
					regMatrix(0,::) := DenseVector(regfactor,0,0,0,0).t 
					regMatrix(1,::) := DenseVector(0,regfactor,0,0,0).t 
					regMatrix(2,::) := DenseVector(0,0,regfactor,0,0).t 
					regMatrix(3,::) := DenseVector(0,0,0,regfactor,0).t 
					regMatrix(4,::) := DenseVector(0,0,0,0,regfactor).t

					//cal sum(yiyit+regMatrix)  //Implementation of step 2 and step 3 and 4
					val userbyItemMat = ratItemVec.map(x => (x._2._2._1,x._2._1*x._2._1.t )).reduceByKey(_+_).map(x=> (x._1,breeze.linalg.pinv(x._2 + regMatrix)))

					// cal sum(rui * yi) where yi is item vectors and rui is the rating. Implementation of step 5
					//org.apache.spark.rdd.RDD[(String, breeze.linalg.DenseVector[Double])] (userid,Densevector)
					val sumruiyi = ratItemVec.map(x => (x._2._2._1,x._2._1 * x._2._2._2.toDouble )).reduceByKey(_+_) 

					// This join will be used in calculating sum yi yit * sum (rui *yi) for each user.
					val joinres = userbyItemMat.join(sumruiyi) 

					// calculates sum(yi*yit) * sum(rui *yi) this gives update of user latent vectors. Combining the results to calculate EQUATION (4)
					myuserMatrix = joinres.map(x=> (x._1,x._2._1 * x._2._2)).partitionBy(new HashPartitioner(10)) 
					//===========================================End of update for myuserMatrix latent vector==========================================================

					//===========================================Homework 4. Implement code to calculate equation 3.===================================================
					//=================You will be required to write code to update myitemMatrix which is the matrix that contains the latent vector for the items
					//Please Fill in your code here.

					val ratUserVec = myuserMatrix.join(ratingByUser.value)

					val regMatrix1 = DenseMatrix.zeros[Double](k,k)  //generate an diagonal matrix with dimension k by k

					regMatrix1(0,::) := DenseVector(regfactor,0,0,0,0).t 
					regMatrix1(1,::) := DenseVector(0,regfactor,0,0,0).t 
					regMatrix1(2,::) := DenseVector(0,0,regfactor,0,0).t 
					regMatrix1(3,::) := DenseVector(0,0,0,regfactor,0).t 
					regMatrix1(4,::) := DenseVector(0,0,0,0,regfactor).t


					val itembyUserMat = ratUserVec.map(x => (x._2._2._1,x._2._1*x._2._1.t )).reduceByKey(_+_).map(x=> (x._1,breeze.linalg.pinv(x._2 + regMatrix1)))
					val sumruixu = ratUserVec.map(x => (x._2._2._1,x._2._1 * x._2._2._2.toDouble )).reduceByKey(_+_) 
					val joinRes = itembyUserMat.join(sumruixu) 
					myitemMatrix = joinRes.map(x=> (x._1,x._2._1 * x._2._2)).partitionBy(new HashPartitioner(10)) 

					//==========================================End of update myitemMatrix latent factor=================================================================
		}
		//======================================================Implement code to recalculate the ratings a user will give an item.====================
		//Hint: This requires multiplying the latent vector of the user with the latent vector of the  item. Please take the input from the command line. and
		// Provide the predicted rating for user 1 and item 914, user 1757 and item 1777, user 1759 and item 231.

		//Your prediction code here
		    if(option.equals("1")){
		      if(latMovieOrUser.equalsIgnoreCase("user_id")){
		        val item_inp = myuserMatrix.lookup(inputId)
				    println("Learned latent vector for User "+inputId+ " : "+ item_inp(0))
		      } 
		      else if(latMovieOrUser.equalsIgnoreCase("movie_id")){
		        val item_inp = myitemMatrix.lookup(inputId)
				    println("Learned latent vector for Movie "+inputId+ " : "+ item_inp(0))
		      }
		    } else if(option.equals("2")){
		     		val user = myuserMatrix.lookup(userInputId)
  				val item = myitemMatrix.lookup(movieInputId)
  				val mult = user(0):*item(0)
  				val sumRes = sum(mult)
  				println("Predicted rating for User "+userInputId+" - Item "+movieInputId+" : "+ sumRes)
		    }
	}
}
