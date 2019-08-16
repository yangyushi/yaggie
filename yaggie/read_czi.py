from xmltodict import parse
from czifile import CziFile, imread
import numpy as np

'''
To install requirements:
pip install xmltodict czifile
'''

class cziReader:
    def __init__(self, file_directory):
        #Initialise reader with file directory
        self.directory     = file_directory
        self.type         = 'Undefined'

    def getStack(self):
        '''
        Get the  image stack from the file,
        rearrange into a more friendly format
        more compatible with the rest of yaggie
        Converts lightsheet data from 16 bit to 8 bit
        '''

        if self.type == 'Series':
            raise Exception('CZI file already defined as a series, if you want to read a stack use cziReader.getStack()')
        self.type = 'Stack'

        #Original dimensions:     [image_no?, ?, c, t, z?, y, x, z?]        
        #Returns as:             [c, t, z, x, y, ?]
        stack            = imread(self.directory)[0][0].astype(np.uint8)
        self.dimensions = stack.shape

        return stack

    def getSerie(self, serie_number):
        '''
        Get the defined image stack from the file,
        rearrange into a more friendly format
        compatible with the rest of yaggie
        Converts lightsheet data from 16 bit to 8 bit
        '''
        if self.type == 'Stack':
            raise Exception('CZI file already defined as a stack, if you want to read a series use cziReader.getSeries(Serie_number)')
        self.type = 'Series'

        serie = imread(self.directory)[serie_number,0,0,0,0].astype(np.uint8)
        self.dimensions = serie.shape
        #Returns as:     [c, t, z, x, y, ?]

        return serie

    def getMetadata(self):
        '''
        Open czi xml header
        convert to dictionary
        extract needed information and convert to correct units
        return new dictionary
        '''

        with CziFile(self.directory) as c:
            cziMetadata = c.metadata()
        cziMetadata = parse(cziMetadata)

        #X, Y, & Z pixel size converted from meters to microns
        x_pixel_size     = (10**6)*float(cziMetadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"])
        y_pixel_size     = (10**6)*float(cziMetadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][1]["Value"])
        z_pixel_size     = (10**6)*float(cziMetadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][2]["Value"])

        voxel_size          = (x_pixel_size * y_pixel_size * z_pixel_size)

        self.metadata     =    {'x_pixel_size'    : x_pixel_size,
                            'y_pixel_size'     : y_pixel_size,
                            'z_pixel_size'     : z_pixel_size,
                            'voxel_size'    : voxel_size}

        return self.metadata


#file_directory     = 'C:/Users/ak18001/tilescan6.czi'
#cziReader         = cziReader(file_directory)
#image             = cziReader.getSerie(2)
#metadata         = cziReader.getMetadata()
#print(metadata)
#print(image)
#print(image.shape)
