'''

@File    :   test.py 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-05 13:18   shenhangke      1.0         None
---------------------
 
'''

import sys

try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print("Successfully imported Spark Modules")
except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

sc = SparkContext("local", "apple")
words = sc.parallelize(["scala", "java", "hadoop", "spark", "akka"])
print(type(words))
print(words.count())
