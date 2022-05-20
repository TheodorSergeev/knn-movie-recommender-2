package test.distributed

import breeze.linalg._
import breeze.numerics._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import shared.predictions._
import test.shared.helpers._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext

class ApproximateTests extends AnyFunSuite with BeforeAndAfterAll {
  
   val separator = "\t"
   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : CSCMatrix[Double] = null
   var test2 : CSCMatrix[Double] = null
   var sc : SparkContext = null

   override def beforeAll {
     train2 = load(train2Path, separator, 943, 1682)
     test2 = load(test2Path, separator, 943, 1682)

     val spark = SparkSession.builder().master("local[2]").getOrCreate();
     spark.sparkContext.setLogLevel("ERROR")
     sc = spark.sparkContext
   }

   // Provide tests to show how to call your code to do the following tasks.
   // Ensure you use the same function calls to produce the JSON outputs in
   // the corresponding application.
   // Add assertions with the answer you expect from your code, up to the 4th
   // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
   test("Approximate kNN predictor with 10 partitions and replication of 2") { 
    var partitionedUsers : Seq[Set[Int]] = partitionUsers(
      943, 
      10, 
      2 
    )

     val top_k = 300
     val approxRes = approxKnnFullPredictionSpark(train2, top_k, sc, partitionedUsers)
     val approxSims = approxRes._1
     val rating_pred = approxRes._2


     val sim_1_1   = approxSims(1 - 1, 1 - 1)
     val sim_1_864 = approxSims(1 - 1, 864 - 1)
     val sim_1_344 = approxSims(1 - 1, 344 - 1)
     val sim_1_16  = approxSims(1 - 1, 16  - 1)
     val sim_1_334 = approxSims(1 - 1, 334 - 1)
     val sim_1_2   = approxSims(1 - 1, 2   - 1)

     val mae = compMatrMAE(test2, rating_pred)


     // Similarity between user 1 and itself
     assert(within(sim_1_1, 0.0, 0.0001))
 
     // Similarity between user 1 and 864
     assert(within(sim_1_864, 0.0, 0.0001))

     // Similarity between user 1 and 344
     assert(within(sim_1_344, 0.2365, 0.0001))

     // Similarity between user 1 and 16
     assert(within(sim_1_16, 0.1857, 0.0001))

     // Similarity between user 1 and 334
     assert(within(sim_1_334, 0.1928, 0.0001))

     // Similarity between user 1 and 2
     assert(within(sim_1_2, 0.0730, 0.0001))

     // MAE on test
     assert(within(mae, 0.7548, 0.0001))
   } 
}
