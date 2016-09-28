
# import necessary packages
import requests
import json
import logging
from StringIO import StringIO
import time
import datetime

############################################################################
# API Document                                                             #
# https://review.udacity.com/api-doc/index.html#!/me/get_me_certifications #
############################################################################

TOKEN = ('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2'
         'VyX2lkIjo0OTc0OCwiZXhwIjoxNDc1NzI0NDIxLCJ0b'
         '2tlbl90eXBlIjoiYXBpIn0.v8h5_sHStP2wWMBkPOUPU'
         'osB2-xH8Qb-LPoR91KqVdc')

CERTIFICATIONS_URL = ('https://review-api.udacity.com/api/v1/me'
                      '/certifications.json')

#'https://review.udacity.com/#!/submissions/dashboard'
headers = {'Authorization': TOKEN, 'Content-Length': '0'}




# Set up a logger thead name, could be used in multi-threading condition
logger = logging.getLogger('Udacity')
logger.setLevel(logging.INFO)
###############################################################################
# Use RotatingFileHandler rather than below
fileName = '{}'.format(str(datetime.date.today()).replace('-',''))
###############################################################################

#fh = logging.handlers.RotatingFileHandler('ud.log', maxBytes=30, backupCount=1)
fh = logging.FileHandler(fileName+'.log')
fh.setLevel(logging.INFO)


ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def normal_case(jsonD):
    print '----------------------------------'
    if jsonD['status']=='certified' :
        for value in jsonD['project']:
            print (value, ':' ,jsonD['project'][value])
    print '----------------------------------'


while True:
    res = requests.get(CERTIFICATIONS_URL, headers=headers)

    jsonDL = json.loads(res.content)

    for jsonD in jsonDL:
    #    normal_case(jsonD)
    # time.sleep(3000)

        # def messages
        mess = '{} waiting [{}], lang_recruit : [{}]'.format(jsonD['project']['name'], jsonD['project']['awaiting_review_count_by_language'], jsonD['project']['languages_to_recruit'])
        if jsonD['project']['awaiting_review_count'] > 0:
            logger.warning(mess)
    print '-------------------------------------------------'

    time.sleep(30)

#jsonD['status']=='certified' and
