import sys
import re
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


wiki_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('count', types.LongType()),
    types.StructField('bytes', types.LongType()),
])


def path_convert(path):
    match = re.search("(\d{8})+(-\d{2})", path)
    return match.group(1) + match.group(2)


def main(in_directory, out_directory):
    wiki = spark.read.csv(in_directory, sep=' ', schema=wiki_schema).withColumn('filename', functions.input_file_name())
    path_convert_udf = functions.udf(path_convert, returnType=types.StringType())
    #print(wiki.limit(1).select("filename").rdd.flatMap(list).collect())
    wiki = wiki.withColumn("dateTime", path_convert_udf(wiki.filename)).cache()

    wiki = wiki.filter(wiki.language == "en")
    wiki = wiki.filter(wiki.title != "Main_Page")
    wiki = wiki.filter(wiki.title.startswith("Special:") == False)
    scores = wiki.groupBy("filename","dateTime" ).agg({'count':'max'})
    scores = scores.select(scores["filename"], scores["max(count)"])
    joined_data = wiki.join(scores, on="filename")
    joined_data = joined_data.filter(joined_data["count"] == joined_data["max(count)"])
    joined_data = joined_data.select(joined_data["dateTime"], joined_data["title"], joined_data["max(count)"])
    joined_data = joined_data.orderBy("dateTime", ascending=False)

    joined_data.write.csv(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
