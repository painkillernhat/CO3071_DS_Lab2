from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
import numpy as np

spark_session = SparkSession.builder.appName("LogAnalysis").getOrCreate()
spark_context = spark_session.sparkContext
log_data = spark_context.textFile("FPT-2018-12-02.log")

def is_valid_record(line):
    fields = line.split(" ")
    criteria = len(fields) == 7 and float(fields[0]) >= 0 and fields[6].isdigit() and int(fields[6]) > 0 and fields[2] != "-"
    return criteria

valid_log_data = log_data.filter(is_valid_record)

# classify service
def get_service_type(line):
    content_name = line.split(" ")[5]
    if content_name.endswith(".mpd") or content_name.endswith(".m3u8"):
        return "HLS"
    elif content_name.endswith(".dash") or content_name.endswith(".ts"):
        return "MPEG-DASH"
    else:
        return "Web Service"

classified_log_data = valid_log_data.map(lambda line: (get_service_type(line), 1))
service_counts = classified_log_data.reduceByKey(lambda a, b: a + b)

for service, count in service_counts.collect():
    print(f"{service}: {count} records")

# extract IP
def extract_ip(line):
    ip = line.split(" ")[1]
    return ip

unique_ips = valid_log_data.map(extract_ip).distinct()

# load IP information
ip_info_data = spark_context.textFile("IPDict.csv")
ip_info_dict = ip_info_data.map(lambda line: (line.split(",")[0], (line.split(",")[1], line.split(",")[2], line.split(",")[3]))).collectAsMap()
ip_info_broadcast = spark_context.broadcast(ip_info_dict)

# enrich log record
def enrich_record(line):
    fields = line.split(" ")
    ip = fields[1]
    additional_info = ip_info_broadcast.value.get(ip, ("Unknown", "Unknown", "Unknown"))
    latency = float(fields[0])
    city = additional_info[1]
    content_size = int(fields[len(fields) - 1])
    return (ip, additional_info, city, latency, fields[5], content_size)

enriched_log_data = valid_log_data.map(enrich_record)

# unique ISPs
unique_isps = enriched_log_data.map(lambda log: log[1][2]).distinct().collect()
print(f"Number of unique ISPs: {len(unique_isps)}")

# records from Ho Chi Minh City
hcm_records = enriched_log_data.filter(lambda log: log[2] == "Ho Chi Minh City")
print(f"Number of records from Ho Chi Minh City: {hcm_records.count()}")

# traffic from Hanoi
hanoi_traffic = enriched_log_data.filter(lambda log: log[2] == "Hanoi").map(lambda log: log[5]).reduce(lambda a, b: a + b)
print(f"Total traffic from Hanoi: {hanoi_traffic}")

# latency statistics
latencies = enriched_log_data.map(lambda log: log[3])
latencies_vector = latencies.map(lambda latency: Vectors.dense(latency))
latency_stats = Statistics.colStats(latencies_vector)

print(f"Mean Latency: {latency_stats.mean()[0]}")
print(f"Maximum Latency: {latency_stats.max()[0]}")
print(f"Minimum Latency: {latency_stats.min()[0]}")

# Additional logic for median, maximum, and second maximum latency
sorted_latencies = latencies.sortBy(lambda latency: latency).collect()

median_index = int((len(sorted_latencies) + 1) / 2) - 1
median_latency = sorted_latencies[median_index]
print(f"Median Latency: {median_latency}")

max_latency_sorted_list = sorted_latencies[-1]
print(f"Maximum Latency (sorted list): {max_latency_sorted_list}")

second_max_latency = sorted_latencies[-2]
print(f"Second Maximum Latency: {second_max_latency}")


spark_session.stop()