package shared

import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import java.security.InvalidParameterException

package object predictions
{
  //================================================================================================
  //===================================== Initial utility code =====================================
  //================================================================================================

  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f: () => Double) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end - start) / 1e6)
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
  //============================================= KNN ==============================================
  //================================================================================================

  // --- Utility and math ---

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


  // --- Averages ---

  def userAvgMap(dataset: CSCMatrix[Double]): DenseVector[Double] = {
    val num_users = dataset.rows
    var sum_usr_rating = DenseVector.zeros[Double](num_users)
    var usr_rating_num = DenseVector.zeros[Double](num_users)

    for ((k,v) <- dataset.activeIterator) {
      // val usr_id = k._1
      sum_usr_rating(k._1) += v
      usr_rating_num(k._1) += 1
    }

    val total_ratings = sum(usr_rating_num)
    val glob_avg = if (total_ratings != 0) sum(sum_usr_rating) / total_ratings else 0

    var user_id = 0

    while (user_id < dataset.rows) {
      if (usr_rating_num(user_id) != 0.0)
        sum_usr_rating(user_id) /= usr_rating_num(user_id)
      else 
        sum_usr_rating(user_id) = glob_avg

      user_id += 1
    }

    sum_usr_rating
  }


  // --- Similarities ---

  // custom matrix multiplication to avoid using breeze - memory problems
  def matrProd(matr1: CSCMatrix[Double], matr2: CSCMatrix[Double]): CSCMatrix[Double] = {
    if (matr1.cols != matr2.rows)
      throw new InvalidParameterException("Matrix sizes are incorrect")
    
    val builder = new CSCMatrix.Builder[Double](rows=matr1.rows, cols=matr2.cols)

    // todo: speed up
    var x = 0
    var y = 0

    while (x < matr2.cols) {
      val mult = matr1 * matr2(0 to matr2.rows - 1, x).toDenseVector 
      // toDenseVector is a major speedup

      while (y < matr1.rows) {
        builder.add(y, x, mult(y))
        y += 1
      }

      y = 0
      x += 1
    }
    
    return builder.result()
    
    // memory issues
    // return matr1 * matr2
  }


  def normalizedDevMatrix(dataset: CSCMatrix[Double], avg_usr_map: DenseVector[Double]): CSCMatrix[Double] = {
    val scaled_rating_builder = new CSCMatrix.Builder[Double](rows=dataset.rows, cols=dataset.cols)

    for ((k,v) <- dataset.activeIterator) {
      scaled_rating_builder.add(k._1, k._2, normalizedDev(dataset(k), avg_usr_map(k._1)))
    }  

    return scaled_rating_builder.result()
  }

  // preprocessing for computation of cosine similarities
  def preprocDataset(scaled_rating: CSCMatrix[Double]): CSCMatrix[Double] = {
    val squared_matrix = scaled_rating *:* scaled_rating       // square the ratings
    val one_vec = DenseVector.ones[Double](scaled_rating.cols) // sum for each user ==
    val denominator = sqrt(squared_matrix * one_vec)           // == sum across each row + take sqrt
    
    // divide numenator by denominator if not zero
    val builder = new CSCMatrix.Builder[Double](rows=scaled_rating.rows, cols=scaled_rating.cols)

    for ((k,v) <- scaled_rating.activeIterator) {
      val denom = denominator(k._1)
      if (denom != 0.0)
        builder.add(k._1, k._2, scaled_rating(k._1, k._2) / denom)
      else
        builder.add(k._1, k._2, 0.0)
    }

    val preproc_dataset = builder.result()
    return preproc_dataset
  }

  // compute cosine similarities and select top k ones for each user
  def computeKnnSimilarities(scaled_rating: CSCMatrix[Double], k: Int): CSCMatrix[Double] = {
    var start = 0.0
    var end = 0.0    

    //start = System.nanoTime() 
    val preproc_dataset = preprocDataset(scaled_rating)   
    //println("preprocess ", (System.nanoTime() - start) / 1e9)

    //start = System.nanoTime() 
    val all_sims = matrProd(preproc_dataset, preproc_dataset.t) // compute cosine similarities
    //println("all sims ", (System.nanoTime() - start) / 1e9)

    //start = System.nanoTime() 
    // zero-out self similarities
    var i = 0
    while (i < preproc_dataset.rows) {
      all_sims(i, i) = 0.0
      i += 1
    }

    // select top k similarities
    var user_id = 0
    val top_sims = new CSCMatrix.Builder[Double](rows=all_sims.rows, cols=all_sims.cols)
    // using builder is much faster than the constructpr

    while (user_id < all_sims.rows) {
      val user_distances = all_sims(0 to all_sims.rows - 1, user_id).toDenseVector

      for (top_neighbor <- argtopk(user_distances, k)) {
        top_sims.add(user_id, top_neighbor, user_distances(top_neighbor))
      }

      user_id += 1
    }
    //println("top k ", (System.nanoTime() - start) / 1e9)

    return top_sims.result()
  }


  // --- Knn ---

  // predict for all users and all items of the test set
  def knnFullPrediction(dataset_train: CSCMatrix[Double], dataset_test: CSCMatrix[Double], k: Int): (CSCMatrix[Double], CSCMatrix[Double]) = {
    var start = 0.0
    var end = 0.0    
    
    // --- user averages --- 
    //start = System.nanoTime() 
    val user_avg_vec = userAvgMap(dataset_train)
    //println("user_avg_vec ", (System.nanoTime() - start) / 1e9)


    // --- similarities --- 
    //start = System.nanoTime() 
    val scaled_rating = normalizedDevMatrix(dataset_train, user_avg_vec) // user-specific weighted-sum deviation matrix
    val top_sims = computeKnnSimilarities(scaled_rating, k)
    //println("top sims     ", (System.nanoTime() - start) / 1e9)

    // --- nominator ---
    //start = System.nanoTime() 
    val nominator = matrProd(top_sims, scaled_rating)
    //println("nominator    ", (System.nanoTime() - start) / 1e9)

    // --- denominator ---
    //start = System.nanoTime() 
    val user_items_if_rated = dataset_train.mapActiveValues(_ => 1.0)
    val denominator = matrProd(abs(top_sims), user_items_if_rated)
    //println("denominator  ", (System.nanoTime() - start) / 1e9)

    // prediction
    //start = System.nanoTime() 
    val builder = new CSCMatrix.Builder[Double](rows=dataset_train.rows, cols=dataset_train.cols)

    var user_id = 0
    var item_id = 0

    while (user_id < dataset_train.rows) {
      while (item_id < dataset_train.cols) {
        val item_dev = 
          if (denominator(user_id, item_id) == 0.0) 
            0.0
          else 
            nominator(user_id, item_id) / denominator(user_id, item_id)
        
        val pred_rating = baselinePrediction(user_avg_vec(user_id), item_dev)

        builder.add(user_id, item_id, pred_rating)

        item_id += 1
      }

      item_id = 0
      user_id += 1
    }

    val pred_test = builder.result()
    //println("builder ", (System.nanoTime() - start) / 1e9)
    return (top_sims, pred_test)
  }

  def compMatrMAE(test: CSCMatrix[Double], pred: CSCMatrix[Double]): Double = {
    var error = 0.0
    var counter = 0

    for ((k,v) <- test.activeIterator) {  // skips items that are not rated in test
      error += scala.math.abs(test(k._1, k._2) - pred(k._1, k._2))
      counter += 1
    }

    return error / counter 
  }



  //================================================================================================
  //============================================= Exact ==============================================
  //================================================================================================



  def parallelKNNComputations(X:CSCMatrix[Double], sc:SparkContext, k:Int):CSCMatrix[Double]= {
    val preproc = preprocDataset(X)
    val br = sc.broadcast(preproc)
    val numUsers = X.rows

    def topK(u: Int): SparseVector[Double]= {
      val preprocBr = br.value
      val kSim = SparseVector[Double](Array.tabulate(numUsers)(_=>0.0))
      val userSim = preprocBr * preprocBr(u, 0 until preprocBr.cols).t.toDenseVector
      userSim(u) = 0
      for (v <- argtopk(userSim, k)) {
        kSim.update(v, userSim(v))
      }
      kSim(u) = 0
      kSim

    }
    val kSims = sc.parallelize(0 until numUsers).map(topK).collect()
    val knn = new CSCMatrix.Builder[Double](rows = numUsers, cols = numUsers)
    for(u <- 0 until numUsers){
      val userKsim = kSims(u)
      for(v <- 0 until numUsers) knn.add(u,v,userKsim(v))
    }
    knn.result()

  }

  def knnFullPredictionSpark(dataset_train: CSCMatrix[Double], dataset_test: CSCMatrix[Double], k: Int, sc: SparkContext): (CSCMatrix[Double], CSCMatrix[Double]) = {
    var start = 0.0
    var end = 0.0

    // --- user averages ---
    //start = System.nanoTime()
    val user_avg_vec = userAvgMap(dataset_train)
    //println("user_avg_vec ", (System.nanoTime() - start) / 1e9)


    // --- similarities ---
    //start = System.nanoTime()
    val scaled_rating = normalizedDevMatrix(dataset_train, user_avg_vec) // user-specific weighted-sum deviation matrix
    val top_sims = parallelKNNComputations(scaled_rating, sc, k)
    //println("top sims     ", (System.nanoTime() - start) / 1e9)

    // --- nominator ---
    //start = System.nanoTime()
    val nominator = matrProd(top_sims, scaled_rating)
    //println("nominator    ", (System.nanoTime() - start) / 1e9)

    // --- denominator ---
    //start = System.nanoTime()
    val user_items_if_rated = dataset_train.mapActiveValues(_ => 1.0)
    val denominator = matrProd(abs(top_sims), user_items_if_rated)
    //println("denominator  ", (System.nanoTime() - start) / 1e9)

    // prediction
    //start = System.nanoTime()
    val builder = new CSCMatrix.Builder[Double](rows=dataset_train.rows, cols=dataset_train.cols)

    var user_id = 0
    var item_id = 0

    while (user_id < dataset_train.rows) {
      while (item_id < dataset_train.cols) {
        val item_dev =
          if (denominator(user_id, item_id) == 0.0)
            0.0
          else
            nominator(user_id, item_id) / denominator(user_id, item_id)

        val pred_rating = baselinePrediction(user_avg_vec(user_id), item_dev)

        builder.add(user_id, item_id, pred_rating)

        item_id += 1
      }

      item_id = 0
      user_id += 1
    }

    val pred_test = builder.result()
    //println("builder ", (System.nanoTime() - start) / 1e9)
    return (top_sims, pred_test)
  }
}
