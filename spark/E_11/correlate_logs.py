import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        n = m.group(1)
        b = m.group(2)
        return Row(host=n, bytes=b)

    else:
        return None

def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    rows = log_lines.map(line_to_row).filter(not_none)
    return rows


    # TODO: return an RDD of Row() objects


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory)).cache()

    # TODO: calculate r.
    logs = logs.groupBy('host').agg({'host':'count', 'bytes':'sum'})
    #logs.show()
    n = logs.count()
    #print(n)
    x = logs.select(functions.sum('count(host)')).collect()[0][0]
    #print(x)
    y = logs.select(functions.sum('sum(bytes)')).collect()[0][0]
    logs = logs.withColumn("xx", logs['count(host)'] * logs['count(host)'])
    xx = logs.select(functions.sum('xx')).collect()[0][0]
    #print(xx)
    logs = logs.withColumn("yy", logs['sum(bytes)'] * logs['sum(bytes)'])
    yy = logs.select(functions.sum('yy')).collect()[0][0]
    logs = logs.withColumn("xy", logs['sum(bytes)'] * logs['count(host)'])
    xy = logs.select(functions.sum('xy')).collect()[0][0]
    #print(xy)
    #logs.show()


    r = ((n * xy) - (x*y))/((math.sqrt((n*xx)-(x*x))) * (math.sqrt((n*yy)-(y*y))))
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
