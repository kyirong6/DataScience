import sys
from pyspark.sql import SparkSession, functions, types
 
spark = SparkSession.builder.appName('example 1').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

data = spark.read.csv('cities.csv', header=True,
                      inferSchema=True)
data.show()