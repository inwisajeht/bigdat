import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.SparkSession

println("############################ Code Starts Here ############################")

def dot (s: String) : Double = {
if (s.contains(".") || s.length == 0) {
return -1
} else {
return s.toDouble
}
}


var df = spark.read
         .format("csv")
         .option("header", "true") //reading the headers
         .option("mode", "DROPMALFORMED")
         .load("caravan-insurance-challenge.csv")

println("Read CSV file and stored as DF. The number of rows in DF is "+df.count)

val colNames = df.schema.names

for( i <- 1 to (colNames.length - 2)){
    val x = colNames(i)
    df = df.withColumn(x, df(x).cast(IntegerType))
    .drop(x+"Tmp")
    .withColumnRenamed(x+"Tmp", x)
}

println("Dataset Loaded and processed")

var trainDF = df.filter( $"ORIGIN".like("train") )
trainDF = trainDF.drop(trainDF.col("ORIGIN"))
var testDF = df.filter( $"ORIGIN".like("test") )
testDF = testDF.drop(testDF.col("ORIGIN"))

println("Train and test Split")

val features = colNames.slice(1, colNames.length - 1)
val label = colNames(colNames.length - 1)

println("Features and labels identified")

val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
var trainDF2 = assembler.transform(trainDF)
var testDF2 = assembler.transform(testDF)

val labelIndexer = new StringIndexer().setInputCol(label).setOutputCol("label")
trainDF2 = labelIndexer.fit(trainDF2).transform(trainDF2)
testDF2 = labelIndexer.fit(testDF2).transform(testDF2)

println("Building Model")

val model = new LogisticRegression().fit(trainDF2)

println("Model Built. Making Predictions.")

val predictions = model.transform(testDF2)

predictions.select ("features", "label", "prediction").show()