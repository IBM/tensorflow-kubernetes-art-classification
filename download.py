import os, errno
from HTMLParser import HTMLParser
import urllib2

ARTS_LIST = 'arts-select.list'
NUMBER_TO_DOWNLOAD = 100     # set to -1 to download all

# The URL for the artifact from Bigquery is a webpage, which contains a link 
# to download the original image.  This class parses for the download link 
class MetArtHTMLParser(HTMLParser):
    # Look for the image link in the http page and download the original image
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            # Look for the keyword selectedOrDefaultDownload in an href
            for attr in attrs:
                if attr[0] == 'href' and 'selectedOrDefaultDownload' in attr[1] :
                    art_url = attr[1].split("'")[1]
                    # Return the URL to download the original image
                    self.data = art_url

with open(ARTS_LIST) as f:
    arts_to_download = f.readlines()
    arts_to_download = [x.strip() for x in arts_to_download]
    f.close()

myparser = MetArtHTMLParser()

for item in arts_to_download:
    # Parse the line to get the culture label and the webpage for the artifact
    pick = item.split("',")
    culture = pick[1].replace(" u'", "")
    webpage = pick[2].replace(" u'", "").replace("')", "")
    print culture, webpage
    
    # Download the webpage and parse for the image URL
    response = urllib2.urlopen(webpage)
    encoding = response.headers.getparam('charset')
    html_page = response.read().decode(encoding)
    
    try:
        myparser.feed(html_page)
    
        # Create a directory with the culture as name if it doesn't exist yet
        # (remove characters that are not valid for directory name)
        culture = culture.replace(",", "")
        culture = culture.replace("/", " ")
        download_dir = 'data/met_art/' + culture
        try: 
            os.makedirs(download_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # Download the image into the directory
        download_path = download_dir + '/' + myparser.data.split("/")[-1]
        image_file = open(download_path, 'wb')
        # Convert the url to the %-encoded format since it may be in other format like utf-8 
        image_url = urllib2.quote(myparser.data.encode(encoding), '/:')
        print "image to download:  ", image_url
        response = urllib2.urlopen(image_url)
        image_file.write(response.read())
        image_file.close()
    except:
        print "Error, skipping url: ", webpage 
    
