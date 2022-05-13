import org.rogach.scallop._
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._
import shared.predictions._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

package scaling {

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val train = opt[String](required = true)
    val test = opt[String](required = true)
    val json = opt[String]()
    val users = opt[Int]()
    val movies = opt[Int]()
    val separator = opt[String](default=Some("\t"))
    val master = opt[String]()
    val num_measurements = opt[Int](default=Some(1))
    verify()
  }

  object Optimizing extends App {
    var conf = new Conf(args)
    // conf object is not serializable, extract values that
    // will be serialized with the parallelize implementations
    val conf_users = conf.users()
    val conf_movies = conf.movies()

    // Remove these lines if encountering/debugging Spark
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = conf.master.toOption match {
      case None => SparkSession.builder().getOrCreate();
      case Some(master) => SparkSession.builder().master(master).getOrCreate();
    }
    spark.sparkContext.setLogLevel("ERROR")
    val sc = spark.sparkContext

    println("Loading training data from: " + conf.train())
    val train = loadSpark(sc, conf.train(), conf.separator(), conf.users(), conf.movies())
    val test  = loadSpark(sc, conf.test (), conf.separator(), conf.users(), conf.movies())


    val top_k = 10
    val large_top_k = 300
    // todo: move the "-1" for breeze indexing somewhere
        

    // compute similarities for individual users
    val preproc_train = preprocDataset(train)
    val knn_similarities = computeKnnSimilarities(preproc_train, top_k)

    val sim_1_1   = knn_similarities(1 - 1, 1 - 1)
    val sim_1_864 = knn_similarities(1 - 1, 864 - 1)
    val sim_1_886 = knn_similarities(1 - 1, 886 - 1)

    println(sim_1_1)
    println(sim_1_864)
    println(sim_1_886)


    // predictions for indivisual users
    val pred_1_1   = knnPrediction(train, top_k, 1 - 1, 1 - 1)
    val pred_327_2 = knnPrediction(train, top_k, 327 - 1, 2 - 1)
  
    println(pred_1_1)
    println(pred_327_2)


    // mae with k = 10
    val mae_10 = compMatrMAE(test, knnFullPrediction(train, test, top_k))
    println(mae_10)


    // measure the speed of prediction and MAE calculation for k = 300
    val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
      compMatrMAE(test, knnFullPrediction(train, test, large_top_k))
    }))
    val timings = measurements.map(t => t._2)
    val mae_300 = measurements(0)._1


    // Save answers as JSON
    def printToFile(content: String, location: String = "./answers.json") =
      Some(new java.io.PrintWriter(location)).foreach{
        f => try{
          f.write(content)
        } finally{ f.close }
      }

    conf.json.toOption match {
      case None => ;
      case Some(jsonFile) => {
        val answers = ujson.Obj(
          "Meta" -> ujson.Obj(
            "train"            -> ujson.Str(conf.train ()),
            "test"             -> ujson.Str(conf.test  ()),
            "users"            -> ujson.Num(conf.users ()),
            "movies"           -> ujson.Num(conf.movies()),
            "master"           -> ujson.Str(conf.master()),
            "num_measurements" -> ujson.Num(conf.num_measurements())
          ),
          "BR.1" -> ujson.Obj(
            "1.k10u1v1"          -> ujson.Num(sim_1_1),
            "2.k10u1v864"        -> ujson.Num(sim_1_864),
            "3.k10u1v886"        -> ujson.Num(sim_1_886),
            "4.PredUser1Item1"   -> ujson.Num(pred_1_1),
            "5.PredUser327Item2" -> ujson.Num(pred_327_2),
            "6.Mae"              -> ujson.Num(mae_10) // todo: or mae_300?
          ),
          "BR.2" ->  ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)), // Datatype of answer: Double
            "stddev (ms)"  -> ujson.Num(std (timings))  // Datatype of answer: Double
          )
        )

        val json = write(answers, 4)

        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }

    println("")
  }

}
