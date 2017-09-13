# Build your data using Google Bigquery and the Met Art collection 


## 1. Install the bigquery client 

You may consider setting this up in a virtualenv:

	virtualenv --system-site-packages .
	source bin/activate
	
Refer to the [instructions](https://cloud.google.com/bigquery/docs/reference/libraries) for Bigquery:

	pip install --upgrade google-cloud-bigquery
	



## 2. Install the Google Cloud SDK

Refer to the [instructions](https://cloud.google.com/sdk/docs/) for the SDK.
For the Mac, download from this site: 

	google-cloud-sdk-168.0.0-darwin-x86_64.tar.gz
	
Unpack this and run the command:  

	./google-cloud-sdk/bin/gcloud init
	
   This will open your browser for you to log into the gmail account and to choose your project in Google cloud:
   
	<your-project-name>
	
   Authenticate for the client by:
   
	./google-cloud-sdk/bin/gcloud auth application-default login
	
## 3. Run a query

Modify the script bigquery.py as appropriate and launch:
	
	python bigquery.py
