// sbt "runMain distributed.Approximate --train data/ml-100k/u2.base --test data/ml-100k/u2.test --json approximate-100k-4-k10-r2.json --k 10 --master local[4] --users 943 --movies 1682 --partitions 10 --replication 2"
import org.rogach.scallop._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._

package distributed {

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val k = opt[Int]()
  val json = opt[String]()
  val users = opt[Int]()
  val movies = opt[Int]()
  val separator = opt[String](default=Some("\t"))
  val replication = opt[Int](default=Some(1))
  val partitions = opt[Int](default=Some(1))
  val master = opt[String]()
  val num_measurements = opt[Int](default=Some(1))
  verify()
}

object Approximate {
  def main(args: Array[String]) {
    var conf = new Conf(args)

    // Remove these lines if encountering/debugging Spark
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = conf.master.toOption match {
      case None => SparkSession.builder().getOrCreate();
      case Some(master) => SparkSession.builder().master(master).getOrCreate();
    }
    val sc = spark.sparkContext

    println("")
    println("******************************************************")

    // conf object is not serializable, extract values that
    // will be serialized with the parallelize implementations
    val conf_users = conf.users()
    val conf_movies = conf.movies()
    val conf_k = conf.k()

    println("Loading training data")
    val train = loadSpark(sc, conf.train(), conf.separator(), conf.users(), conf.movies())
    val test = loadSpark(sc, conf.test(), conf.separator(), conf.users(), conf.movies())
    var knn : CSCMatrix[Double] = null

    println("Partitioning users")
    var partitionedUsers : Seq[Set[Int]] = partitionUsers(
      conf.users(), 
      conf.partitions(), 
      conf.replication()
    )

    val top_k = 10
    val approxRes = tmp_approxKnnFullPredictionSpark(train, test, top_k, sc, conf.partitions(), conf.replication())
    val approxSims = approxRes._1
    val approxPreds = approxRes._2

    
    val sim_1_1   = approxSims(1 - 1, 1 - 1)
    val sim_1_864 = approxSims(1 - 1, 864 - 1)
    val sim_1_344 = approxSims(1 - 1, 344 - 1)
    val sim_1_16  = approxSims(1 - 1, 16  - 1)
    val sim_1_334 = approxSims(1 - 1, 334 - 1)
    val sim_1_2   = approxSims(1 - 1, 2   - 1)

    println(sim_1_1)
    println(sim_1_864)
    println(sim_1_344)
    println(sim_1_16)
    println(sim_1_334)
    println(sim_1_2)

    val mae_10 = compMatrMAE(test, approxPreds)
    println(mae_10)

    /*
    val measurements = (1 to scala.math.max(1,conf.num_measurements()))
      .map(_ => timingInMs( () => {
      // Use partitionedUsers here
      0.0
    }))
    val mae = measurements(0)._1
    val timings = measurements.map(_._2)
    
    // Save answers as JSON
    def printToFile(content: String,
                    location: String = "./answers.json") =
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
            "train" -> ujson.Str(conf.train()),
            "test" -> ujson.Str(conf.test()),
            "k" -> ujson.Num(conf.k()),
            "users" -> ujson.Num(conf.users()),
            "movies" -> ujson.Num(conf.movies()),
            "master" -> ujson.Str(sc.getConf.get("spark.master")),
            "num-executors" -> ujson.Str(if (sc.getConf.contains("spark.executor.instances"))
                                            sc.getConf.get("spark.executor.instances")
                                         else
                                            ""),
            "num_measurements" -> ujson.Num(conf.num_measurements()),
            "partitions" -> ujson.Num(conf.partitions()),
            "replication" -> ujson.Num(conf.replication()) 
          ),
          "AK.1" -> ujson.Obj(
            "knn_u1v1" -> ujson.Num(0.0),
            "knn_u1v864" -> ujson.Num(0.0),
            "knn_u1v344" -> ujson.Num(0.0),
            "knn_u1v16" -> ujson.Num(0.0),
            "knn_u1v334" -> ujson.Num(0.0),
            "knn_u1v2" -> ujson.Num(0.0)
          ),
          "AK.2" -> ujson.Obj(
            "mae" -> ujson.Num(mae) 
          ),
          "AK.3" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)),
            "stddev (ms)" -> ujson.Num(std(timings))
          )
        )
        val json = write(answers, 4)

        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }
    */

    println("")
    spark.stop()
  } 

}

}
