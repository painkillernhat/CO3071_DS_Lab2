from pyspark.sql import SparkSession
from datetime import datetime
import pytz

spark_session = SparkSession.builder.appName("Log Processing").getOrCreate()
log_data = spark_session.sparkContext.textFile("FPT-2018-12-02.log")

# filter correct records
def filter_correct_records(line):
    fields = line.split(" ")
    criteria = len(fields) == 7 and float(fields[0]) >= 0 and fields[6].isdigit() and int(fields[6]) > 0 and fields[2] != "-"
    return criteria

# convert time format
def convert_time(line):
    fields = line.split(" ")
    input_data = fields[3] + " " + fields[4]
    time_format = "[%d/%b/%Y:%H:%M:%S %z]"
    try:
        timestamp = datetime.strptime(input_data, time_format).replace(tzinfo=pytz.UTC).timestamp()
        return timestamp
    except ValueError:
        return None

# filter correct records
filtered_data = log_data.filter(filter_correct_records)
filtered_data = filtered_data.filter(lambda x: convert_time(x) is not None)
sorted_data = filtered_data.sortBy(convert_time)
print("Top 10 correct records:")
for record in sorted_data.take(10):
    print(record)

# filter incorrect records
def filter_incorrect_records(line):
    return not filter_correct_records(line)

# filter incorrect records
filtered_fail_data = log_data.filter(filter_incorrect_records)
filtered_fail_data = filtered_fail_data.filter(lambda x: convert_time(x) is not None)
sorted_fail_data = filtered_fail_data.sortBy(convert_time)
print("\nTop 10 incorrect records:")
for record in sorted_fail_data.take(10):
    print(record)

# stop spark session
spark_session.stop()