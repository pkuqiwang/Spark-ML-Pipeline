# Spark-ML-Pipeline
```
%spark
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType}

//calculate minuted from midnight, input is military time format 
def getMinuteOfDay(depTime: String) : Int = (depTime.toInt / 100).toInt * 60 + (depTime.toInt % 100)

//define schema for raw data
case class Flight(Month: String, DayofMonth: String, DayOfWeek: String,	DepTime: Int, CRSDepTime: Int, ArrTime: Int, CRSArrTime: Int, UniqueCarrier: String, ActualElapsedTime: Int, CRSElapsedTime: Int, AirTime: Int, ArrDelay: Double, DepDelay: Int, Origin: String, Distance: Int)

val flight2007 = sc.textFile("/tmp/airflightsdelays/flights_2007.csv.bz2")
val header = flight2007.first

val trainingData = flight2007
                    .filter(x => x != header)
                    .map(x => x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(17) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(p => Flight(p(1), p(2), p(3), getMinuteOfDay(p(4)), getMinuteOfDay(p(5)), getMinuteOfDay(p(6)), getMinuteOfDay(p(7)), p(8), p(11).toInt, p(12).toInt, p(13).toInt, p(14).toDouble, p(15).toInt, p(16), p(18).toInt))
                    .toDF
                    
trainingData.cache

val flight2008 = sc.textFile("/tmp/airflightsdelays/flights_2008.csv.bz2")
val testingData = flight2008
                    .filter(x => x != header)
                    .map(x => x.split(","))
                    .filter(x => x(21) == "0")
                    .filter(x => x(17) == "ORD")
                    .filter(x => x(14) != "NA")
                    .map(p => Flight(p(1), p(2), p(3), getMinuteOfDay(p(4)), getMinuteOfDay(p(5)), getMinuteOfDay(p(6)), getMinuteOfDay(p(7)), p(8), p(11).toInt, p(12).toInt, p(13).toInt, p(14).toDouble, p(15).toInt, p(16), p(18).toInt))
                    .toDF
                    
testingData.cache
```

```
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

//tranformor to convert string to category values
val monthIndexer = new StringIndexer().setInputCol("Month").setOutputCol("MonthCat")
val DayofMonthIndexer = new StringIndexer().setInputCol("DayofMonth").setOutputCol("DayofMonthCat")
val DayOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekCat")
val UniqueCarrierIndexer = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierCat")
val OriginIndexer = new StringIndexer().setInputCol("Origin").setOutputCol("OriginCat")

//assemble raw feature
val assembler = new VectorAssembler()
                    .setInputCols(Array("MonthCat", "DayofMonthCat", "DayOfWeekCat", "UniqueCarrierCat", "OriginCat", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime", "AirTime","DepDelay", "Distance"))
                    .setOutputCol("rawFeatures")
```

```
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StandardScaler

//vestor slicer
val slicer = new VectorSlicer().setInputCol("rawFeatures").setOutputCol("slicedfeatures").setNames(Array("MonthCat", "DayofMonthCat", "DayOfWeekCat", "UniqueCarrierCat", "DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "DepDelay", "Distance"))

//scale the features
val scaler = new StandardScaler().setInputCol("slicedfeatures").setOutputCol("features").setWithStd(true).setWithMean(true)

//labels for binary classifier
val binarizerClassifier = new Binarizer().setInputCol("ArrDelay").setOutputCol("binaryLabel").setThreshold(15.0)

//logistic regression
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("binaryLabel").setFeaturesCol("features")

// Chain indexers and tree in a Pipeline
val lrPipeline = new Pipeline().setStages(Array(monthIndexer, DayofMonthIndexer, DayOfWeekIndexer, UniqueCarrierIndexer, OriginIndexer, assembler, slicer, scaler, binarizerClassifier, lr))

// Train model. 
val lrModel = lrPipeline.fit(trainingData)

// Make predictions.
val lrPredictions = lrModel.transform(testingData)

// Select example rows to display.
lrPredictions.select("prediction", "binaryLabel", "features").show(20)
```

```
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.PCA

//index category index in raw feature
val indexer = new VectorIndexer().setInputCol("rawFeatures").setOutputCol("rawFeaturesIndexed").setMaxCategories(10)

//PCA
val pca = new PCA().setInputCol("rawFeaturesIndexed").setOutputCol("features").setK(10)

//label for multi class classifier
val bucketizer = new Bucketizer().setInputCol("ArrDelay").setOutputCol("multiClassLabel").setSplits(Array(Double.NegativeInfinity, 0.0, 15.0, Double.PositiveInfinity))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("multiClassLabel").setFeaturesCol("features")

// Chain all into a Pipeline
val dtPipeline = new Pipeline().setStages(Array(monthIndexer, DayofMonthIndexer, DayOfWeekIndexer, UniqueCarrierIndexer, OriginIndexer, assembler, indexer, pca, bucketizer, dt))

// Train model. 
val dtModel = dtPipeline.fit(trainingData)

// Make predictions.
val dtPredictions = dtModel.transform(testingData)

// Select example rows to display.
dtPredictions.select("prediction", "multiClassLabel", "features").show(20)
```
