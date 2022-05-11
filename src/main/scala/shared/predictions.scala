package shared

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext

package object predictions
{
  //================================================================================================
  //===================================== Initial utility code =====================================
  //================================================================================================

  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f : ()=>Double ) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end-start)/1000000.0)
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def mean(s :Seq[Double]): Double =  if (s.size > 0) s.reduce(_+_) / s.length else 0.0

  def std(s :Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else { 
      val m = mean(s)
      scala.math.sqrt(s.map(x => scala.math.pow(m-x, 2)).sum / s.length.toDouble) 
    }
  }


  def load(path : String, sep : String, nbUsers : Int, nbMovies : Int) : CSCMatrix[Double] = {
    val file = Source.fromFile(path)
    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbMovies) 
    for (line <- file.getLines) {
      val cols = line.split(sep).map(_.trim)
      toInt(cols(0)) match {
        case Some(_) => builder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
        case None => None
      }
    }
    file.close
    builder.result()
  }

  def loadSpark(sc: org.apache.spark.SparkContext,  path: String, sep: String, nbUsers: Int, nbMovies: Int): CSCMatrix[Double] = {
    val file = sc.textFile(path)
    val ratings = file
      .map(l => {
        val cols = l.split(sep).map(_.trim)
        toInt(cols(0)) match {
          case Some(_) => Some(((cols(0).toInt-1, cols(1).toInt-1), cols(2).toDouble))
          case None => None
        }
      })
      .filter({ case Some(_) => true
                 case None => false })
      .map({ case Some(x) => x
             case None => ((-1, -1), -1) }).collect()

    val builder = new CSCMatrix.Builder[Double](rows=nbUsers, cols=nbMovies)
    for ((k,v) <- ratings) {
      v match {
        case d: Double => {
          val u = k._1
          val i = k._2
          builder.add(u, i, d)
        }
      }
    }
    return builder.result
  }

  def partitionUsers (nbUsers : Int, nbPartitions : Int, replication : Int) : Seq[Set[Int]] = {
    val r = new scala.util.Random(1337)
    val bins : Map[Int, collection.mutable.ListBuffer[Int]] = (0 to (nbPartitions-1))
       .map(p => (p -> collection.mutable.ListBuffer[Int]())).toMap
    (0 to (nbUsers-1)).foreach(u => {
      val assignedBins = r.shuffle(0 to (nbPartitions-1)).take(replication)
      for (b <- assignedBins) {
        bins(b) += u
      }
    })
    bins.values.toSeq.map(_.toSet)
  }


  //================================================================================================
  //========================================= Custom types =========================================
  //================================================================================================

  type RatingArr = Array[Rating]
  type DistrRatingArr = org.apache.spark.rdd.RDD[Rating]

  type RatingPredFunc = (Int, Int) => Double
  type SimilarityFunc = (Int, Int, RatingArr) => Double

  type TrainerOfPredictor = (RatingArr) => RatingPredFunc
  type DistrTrainerOfPredictor = (DistrRatingArr) => RatingPredFunc


   

  //================================================================================================
  //=========================================== Baseline ===========================================
  //================================================================================================

  // utility and math

  def scaleRatingToUserAvg(rating: Double, avgRating: Double): Double = {
    if (rating > avgRating)
      5.0 - avgRating
    else if (rating < avgRating)
      avgRating - 1.0
    else
      1.0
  }

  def normalizedDev(rating: Double, avg_user_rating: Double): Double = {
    return (rating - avg_user_rating) / scaleRatingToUserAvg(rating, avg_user_rating)
  }

  def baselinePrediction(user_avg: Double, item_dev: Double): Double = {
    return user_avg + item_dev * scaleRatingToUserAvg(user_avg + item_dev, user_avg)
  }

  def sqDev(rating: Double, usr_avg: Double): Double = {
    // squared normalized deviation
    val dev = normalizedDev(rating, usr_avg)
    return dev * dev
  }


  // some averages

  def globalAvgRating(dataset: CSCMatrix[Double]): Double = {
    var cumsum = 0.0
    var counter = 0.0

    for ((k,v) <- dataset.activeIterator) {
      cumsum += v
      counter += 1
    }  

    if (counter == 0.0)
      0.0
    else
      cumsum / counter
  }

  def userAvgMap(dataset: CSCMatrix[Double]): DenseVector[Double] = {
    val glob_avg = globalAvgRating(dataset)

    val num_users = dataset.rows
    var sum_usr_rating = DenseVector.zeros[Double](num_users)
    var usr_rating_num = DenseVector.zeros[Double](num_users)

    for ((k,v) <- dataset.activeIterator) {
      // val usr_id = k._1
      sum_usr_rating(k._1) += v
      usr_rating_num(k._1) += 1
    }  

    sum_usr_rating /:/ usr_rating_num
  }


  // similarity computation
  
  def normalizedDevMatrix(dataset: CSCMatrix[Double]): CSCMatrix[Double] = {
    val avg_usr_map = userAvgMap(dataset)

    // center and scale
    val scaled_rating_builder = new CSCMatrix.Builder[Double](rows=dataset.rows, cols=dataset.cols)

    for ((k,v) <- dataset.activeIterator) {
      // val usr_id = k._1
      scaled_rating_builder.add(k._1, k._2, normalizedDev(dataset(k._1, k._2), avg_usr_map(k._1)))
    }  

    return scaled_rating_builder.result()
  }

  // preprocessing for computation of cosine similarities
  def preprocDataset(dataset: CSCMatrix[Double]): CSCMatrix[Double] = {
    val num_users = dataset.rows
        
    val scaled_rating = normalizedDevMatrix(dataset)

    // normalize
    val squared_matrix = scaled_rating *:* scaled_rating // square the ratings
    val one_vec = DenseVector.ones[Double](dataset.cols) // sum for each user ==
    val denominator = squared_matrix * one_vec           // == sum across each row
    sqrt.inPlace(denominator) // take sqrt of the sum to get the final denominators
    
    val builder = new CSCMatrix.Builder[Double](rows=dataset.rows, cols=dataset.cols)

    for ((k,v) <- dataset.activeIterator) {
      // val usr_id = k._1
      val denom = denominator(k._1)
      if (denom != 0.0)
        builder.add(k._1, k._2, denom)
      else
        builder.add(k._1, k._2, 0.0)
    }

    val preproc_dataset = builder.result()
    return preproc_dataset
  }

  // compute cosine similarities
  def computeSimilarities(preproc_dataset: CSCMatrix[Double]): CSCMatrix[Double] = {    
    return preproc_dataset * preproc_dataset.t
  }

  // compute cosine similarities and select top k ones for each user
  def computeKnnSimilarities(preproc_dataset: CSCMatrix[Double], k: Int): CSCMatrix[Double] = {    
    val all_sims = computeSimilarities(preproc_dataset)
    
    val top_sims = CSCMatrix.zeros[Double](all_sims.rows, all_sims.cols)

    for (user_id <- 0 to all_sims.rows - 1) {
      val user_distances = all_sims(0 to all_sims.rows - 1, user_id)

      for (top_neighbor <- argtopk(user_distances, k)) {
        top_sims(user_id, top_neighbor) = user_distances(top_neighbor)
      }
    }

    return top_sims
  }


  // k-NN functions

  // compute k-NN similarities for single user with id = userId
  def userKnn(dataset: CSCMatrix[Double], k: Int, user_id: Int): (DenseVector[Int], DenseVector[Double]) = {
    // compute all similarities
    val preproc_arr = preprocDataset(dataset)
    val distances = computeSimilarities(preproc_arr)
    val user_distances = distances(0 to dataset.rows - 1, user_id)

    // select top k nearest neighbours for each user
    val top_neighbors = DenseVector.zeros[Int](k)
    var pos = 0

    for (i <- argtopk(user_distances, k)) {
      top_neighbors(pos) = i
      pos += 1
    }

    // make new similarities matrix
    val similarities = DenseVector.zeros[Double](dataset.rows)

    for (i <- top_neighbors) {
      similarities(i) = distances(user_id, i)
    }

    return (top_neighbors, similarities)
  }

  // make a prediction for a single user on a single item
  def knnPrediction(dataset: CSCMatrix[Double], k: Int, user_id: Int, item_id: Int): Double = {
    val preproc_dataset = preprocDataset(dataset)
    val top_sims = computeKnnSimilarities(preproc_dataset, k)
    
    // user-specific weighted-sum deviation matrix
    val scaled_matrix = normalizedDevMatrix(dataset)
    val nominator = top_sims * scaled_matrix
  
    val user_items_if_rated = CSCMatrix.zeros[Double](dataset.rows, dataset.cols)
    for ((k,v) <- dataset.activeIterator) {
      if (dataset(k._1, k._2) != 0.0)
        user_items_if_rated(k._1, k._2) = 1.0
    }
    val denominator = abs(top_sims) * user_items_if_rated

    //val item_dev = nominator /:/ denominator

    // prediction
    val item_dev = 
      if (denominator(user_id, item_id) == 0.0) 
        0.0 
      else 
        nominator(user_id, item_id) / denominator(user_id, item_id)

    val user_avg_vec = userAvgMap(dataset)
    val pred_rating = baselinePrediction(user_avg_vec(user_id), item_dev)

    return pred_rating
  }

  // predict knn on multiple users
  def knnFullPrediction(dataset_train: CSCMatrix[Double], dataset_test: CSCMatrix[Double], k: Int): CSCMatrix[Double] = {
    val preproc_dataset = preprocDataset(dataset_train)
    val top_sims = computeKnnSimilarities(preproc_dataset, k)
    
    // user-specific weighted-sum deviation matrix
    val scaled_matrix = normalizedDevMatrix(dataset_train)
    val nominator = top_sims * scaled_matrix
  
    val user_items_if_rated = CSCMatrix.zeros[Double](dataset_train.rows, dataset_train.cols)
    for ((k,v) <- dataset_train.activeIterator) {
      if (dataset_train(k._1, k._2) != 0.0)
        user_items_if_rated(k._1, k._2) = 1.0
    }
    val denominator = abs(top_sims) * user_items_if_rated

    // prediction
    val user_avg_vec = userAvgMap(dataset_train)

    val builder = new CSCMatrix.Builder[Double](rows=dataset_train.rows, cols=dataset_train.cols)
    for ((k,v) <- dataset_test.activeIterator) {
      val user_id = k._1
      val item_id = k._2

      if (dataset_test(user_id, item_id) == 0.0) 
        builder.add(user_id, item_id, 0.0)
      else {
        val item_dev = 
          if (denominator(user_id, item_id) == 0.0) 
            0.0
          else 
            nominator(user_id, item_id) / denominator(user_id, item_id)
        
        val pred_rating = baselinePrediction(user_avg_vec(user_id), item_dev)

        builder.add(user_id, item_id, pred_rating)
      }
        
    }

    val pred_test = builder.result()

    return pred_test
  }

  def compMatrMAE(real: CSCMatrix[Double], pred: CSCMatrix[Double]): Double = {
    return globalAvgRating(abs(real - pred))
  }

  /*
  // recommend top movies based on k-NN
  def recommend(dataset: CSCMatrix[Double], k: Int, user_id: Int): DenseVector[Int] = {
    // dataset contains all users from train and test?

    val user_avg_vec = userAvgMap(dataset)

    // normalized rating deviation
    val norm_rating_dev_builder = new CSCMatrix.Builder[Double](rows=dataset.rows, cols=dataset.cols)
    val user_item_if_rated_builder = new CSCMatrix.Builder[Double](rows=dataset.rows, cols=dataset.cols)

    var new_el = 0.0

    for ((k,v) <- dataset.activeIterator) {
      norm_rating_dev_builder.add(k._1, k._2, 
        normalizedDev(dataset(k._1, k._2), user_avg_vec(k._1))
      )

      if (dataset(k._1, k._2) != 0)
        new_el = 1.0
      else
        new_el = 0.0

      user_item_if_rated_builder.add(k._1, k._2, new_el)
    }  

    val norm_rating_dev    = norm_rating_dev_builder   .result()
    val user_item_if_rated = user_item_if_rated_builder.result()

    // similarities
    val user_similarities = similaritiesKnn(dataset, k)._2

    // user-specific weighted-sum deviation for item i
    val numenator   = user_similarities * norm_rating_dev
    var denominator = user_similarities * user_item_if_rated
    val item_rating_dev = numenator /:/ denominator

    // personalized rating
    val predictions = user_avg_vec + item_rating_dev * scaled2_rating

    return predictions
  }*/
}