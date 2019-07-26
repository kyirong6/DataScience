import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

schema = types.StructType([ # commented-out fields won't be read
    #types.StructField('archived', types.BooleanType(), False),
    types.StructField('author', types.StringType(), False),
    #types.StructField('author_flair_css_class', types.StringType(), False),
    #types.StructField('author_flair_text', types.StringType(), False),
    #types.StructField('body', types.StringType(), False),
    #types.StructField('controversiality', types.LongType(), False),
    #types.StructField('created_utc', types.StringType(), False),
    #types.StructField('distinguished', types.StringType(), False),
    #types.StructField('downs', types.LongType(), False),
    #types.StructField('edited', types.StringType(), False),
    #types.StructField('gilded', types.LongType(), False),
    types.StructField('id', types.StringType(), False),
    #types.StructField('link_id', types.StringType(), False),
    #types.StructField('name', types.StringType(), False),
    #types.StructField('parent_id', types.StringType(), True),
    #types.StructField('retrieved_on', types.LongType(), False),
    types.StructField('score', types.LongType(), False),
    #types.StructField('score_hidden', types.BooleanType(), False),
    types.StructField('subreddit', types.StringType(), False),
    #types.StructField('subreddit_id', types.StringType(), False),
    #types.StructField('ups', types.LongType(), False),
])


def main(in_directory, out_directory):
    # working but not optimal!

    comments = spark.read.json(in_directory, schema=schema).cache()

    # TODO
    averages = comments.groupBy('subreddit').agg({'score':'avg'})
    averages = averages.filter(averages['avg(score)'] > 0).cache()
    averages = functions.broadcast(averages)

    #averages.show()
    joined_data = comments.join(averages, on="subreddit")
    joined_data = joined_data.withColumn("rel_score", joined_data.score/joined_data["avg(score)"])
    max = joined_data.groupBy('subreddit').agg({'rel_score':'max'})
    max = functions.broadcast(max)
    joined_data = joined_data.join(max, on="subreddit")
    joined_data = joined_data.filter(joined_data["rel_score"] == joined_data["max(rel_score)"]).cache()
    joined_data = joined_data.select(joined_data['subreddit'], joined_data['author'], joined_data['rel_score'])
    #joined_data.show(100)
    #joined_data.show(500)


    joined_data.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
