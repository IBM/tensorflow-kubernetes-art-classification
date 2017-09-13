

"""Simple application that performs a query with BigQuery."""

import uuid

from google.cloud import bigquery


def query_metart():
    # Specify your Google Cloud project to connect to
    client = bigquery.Client(project="booming-flash-176918")

    query_job = client.run_async_query(str(uuid.uuid4()), """
        #standardSQL
        SELECT culture, COUNT(*) c 
        FROM `bigquery-public-data.the_met.objects`
        GROUP BY 1
        ORDER BY c DESC""")


    query_job.begin()
    query_job.result()  # Wait for job to complete.

    destination_table = query_job.destination
    destination_table.reload()
    for row in destination_table.fetch_data():
        print(row)


if __name__ == '__main__':
    query_metart()
# [END all]