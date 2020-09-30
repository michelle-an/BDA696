#!/Users/michellean/workspace/courses/bda696/src/BDA696/.venv/bin/python3

import sys

import mariadb
from pyspark import StorageLevel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, IntegerType, StructField, StructType


def printout(text):
    print(f"*** {text} ***")
    x = len(text) + 8
    print("-" * x)


def main():
    # create spark session
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.catalog.clearCache()

    # Connect to MariaDB Platform
    try:
        connection = mariadb.connect(
            user="root",
            password="michelle123",  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="baseball",
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)

    printout("success connecting...")

    # setup schema
    schema = StructType(
        [
            StructField(name="game_id", dataType=IntegerType(), nullable=True),
            StructField(name="batter", dataType=IntegerType(), nullable=True),
            StructField(name="hit", dataType=IntegerType(), nullable=True),
            StructField(name="atbat", dataType=IntegerType(), nullable=True),
            StructField(name="local_date", dataType=DateType(), nullable=True),
        ]
    )

    # create empty spark dataframe using schema
    df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

    # import batter counts table and game table
    cursor = connection.cursor()
    count = 0
    printout("creating table...")
    cursor.execute(
        f"SELECT bc.game_id, bc.batter, bc.Hit, bc.atbat, gt.local_date \
        FROM batter_counts bc INNER JOIN game_temp gt on bc.game_id = gt.game_id ORDER BY game_id"
    )
    printout("importing table...")
    for (game_id, batter, hit, atbat, local_date) in cursor:
        to_insert = spark.createDataFrame(
            [
                (game_id, batter, hit, atbat, local_date),
            ]
        )
        df = df.union(to_insert)
        count += 1
        if count % 500 == 0:
            print(f"\timporting row {count}...")
    print(df.show(n=200))
    df.createOrReplaceTempView("rolling_avg_temp")
    df.persist(StorageLevel.MEMORY_AND_DISK)

    # solve for rolling batting averages
    printout("solving for rolling batting averages...")
    rolling_df = spark.sql(
        f"""SELECT rat1.batter, SUM(rat2.Hit) AS sum_hits , SUM(rat2.atbat) AS sum_bats \
        FROM rolling_avg_temp rat1 JOIN rolling_avg_temp rat2 ON rat2.local_date \
        BETWEEN DATE_ADD(rat1.local_date, - 100) AND rat1.local_date AND \
        rat1.batter = rat2.batter GROUP BY rat1.batter"""
    )

    print(rolling_df.show(n=20))
    rolling_df.createOrReplaceTempView("rolling_df")
    rolling_df.persist(StorageLevel.MEMORY_AND_DISK)

    # create array column of all necessary data
    printout("converting data to array...")
    rolling_df = spark.sql(
        """SELECT * , SPLIT(CONCAT(CASE WHEN batter IS NULL THEN "" \
        ELSE batter END, " ", CASE WHEN sum_hits IS NULL OR sum_bats IS NULL THEN "" \
        ELSE ROUND(sum_hits/sum_bats, 3) END), " ") \
        AS array_with_rolling_averages FROM rolling_df"""
    )
    print(rolling_df.show(n=20))

    # fit array column to count vectorizer
    printout("running vectorizer and transformer...")
    count_vectorizer = CountVectorizer(
        inputCol="array_with_rolling_averages", outputCol="array_vector"
    )
    count_vectorizer_fitted = count_vectorizer.fit(rolling_df)

    # transform the fitted count vectorizer
    rolling_df = count_vectorizer_fitted.transform(rolling_df)
    print(rolling_df.show(n=20, truncate=False))

    return


if __name__ == "__main__":
    sys.exit(main())
