import org.rogach.scallop._
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import ujson._

package economics {

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val json = opt[String]()
    verify()
  }

  object Economics {
    def main(args: Array[String]) {
      println("")
      println("******************************************************")

      var conf = new Conf(args)

      val ICCM7_buying_cost = 38600
      val ICCM7_renting_cost = 20.40
      val RPi_buying_cost = 108.48
      val RPi_RAM = 8
      val ICCM7_RAM = 24 * 64
      val ICCM7_CPU = 2 * 14 * 2.6
      val RPi_CPU = 4 * 1.5
      val Container_RAM_RPi_Daily = 0.00000016 * 86400 * 4 * RPi_RAM
      val Container_CPU_RPi_Daily = 0.00000114 * 86400
      val x4RPi_Daily_Idle = 4 * 0.003 * 0.25 * 24
      val x4RPi_Daily_Computing = 4 * 0.004 * 0.25 * 24

      val days_less_expensive = math.ceil(ICCM7_buying_cost / ICCM7_renting_cost)

      val Container_Daily_Cost = Container_RAM_RPi_Daily + Container_CPU_RPi_Daily
      val MinRentingDaysIdleRPiPower = math.round((4 * RPi_buying_cost)/(Container_Daily_Cost - x4RPi_Daily_Idle)) //here guys use ceil, but in task it's said "(Round up to the nearest integer in each case)"
      val MinRentingDaysComputingRPiPower = math.round((4 * RPi_buying_cost)/(Container_Daily_Cost - x4RPi_Daily_Computing)) //here guys use ceil, but in task it's said "(Round up to the nearest integer in each case)"

      val RPi_equal_ICCM7 = math.floor(ICCM7_buying_cost/RPi_buying_cost)
      val RPi_RAM_ICCM7 = (RPi_equal_ICCM7 * RPi_RAM) / ICCM7_RAM
      val RPi_CPU_ICCM7 = (RPi_equal_ICCM7 * RPi_CPU) / ICCM7_CPU //Or rpi_equal/4/28 


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
            "E.1" -> ujson.Obj(
              "MinRentingDays" -> ujson.Num(days_less_expensive) // Datatype of answer: Double
            ),
            "E.2" -> ujson.Obj(
              "ContainerDailyCost" -> ujson.Num(Container_Daily_Cost),
              "4RPisDailyCostIdle" -> ujson.Num(x4RPi_Daily_Idle),
              "4RPisDailyCostComputing" -> ujson.Num(x4RPi_Daily_Computing),
              "MinRentingDaysIdleRPiPower" -> ujson.Num(MinRentingDaysIdleRPiPower),
              "MinRentingDaysComputingRPiPower" -> ujson.Num(MinRentingDaysComputingRPiPower)
            ),
            "E.3" -> ujson.Obj(
              "NbRPisEqBuyingICCM7" -> ujson.Num(RPi_equal_ICCM7),
              "RatioRAMRPisVsICCM7" -> ujson.Num(RPi_RAM_ICCM7),
              "RatioComputeRPisVsICCM7" -> ujson.Num(RPi_CPU_ICCM7)
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

}
