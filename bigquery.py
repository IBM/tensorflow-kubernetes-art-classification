
""" Query the BigQuery public table for the Metropolitan Art. """

import uuid

from google.cloud import bigquery


def query_metart():
    # Specify your Google Cloud project to connect to
    client = bigquery.Client(project="booming-flash-176918")

    query_job = client.query("""
        #standardSQL
        SELECT department, culture, link_resource
        FROM `bigquery-public-data.the_met.objects`
        WHERE culture IS NOT NULL""")

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print(row[0:3])

if __name__ == '__main__':
    query_metart()
# [END all]
