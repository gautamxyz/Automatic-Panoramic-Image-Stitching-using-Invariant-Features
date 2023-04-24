# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="panorama.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)