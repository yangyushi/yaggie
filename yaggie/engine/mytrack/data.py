from xml.etree import ElementTree as et
from io import StringIO
try:
    from read import read_pkl, read_xml, read_meta
except:
    from .read import read_pkl, read_xml, read_meta
import numpy as np

class Metadata(dict):
    def __init__(self, xml):
        """
        structure:
            self ---> {img_name: img_content, ...}
            img_content ---> {channel_name: channel_content, ...}
            channel_content ---> [z1, z2, ..., zn] (numpy array)
        pixel_information:
            self.pixel_info ---> {image_name: image_pixel_info, ...}
            image_pixel_info ---> {pixel_number, pixel_size, ...}
        """
        self.root = self.__simple_parse(xml)
        self.pixel_information = {}
        self.read_images()

    def read_images(self):
        images = self.root.findall('Image')
        for img in images:
            name = img.get('ID')
            image_content = self.get_image_content(img)
            self.update({name: image_content})
            self.pixel_information.update({name: self.get_pixel_info(img.find('Pixels'))})
    
    def get_image_content(self, image):
        image_content = {}
        pixels = image.find('Pixels')
        for channel in pixels.findall('Channel'):
            name = channel.get('ID')
            channel_id = name.split(':')[-1]
            channel_content = self.get_channel_content(pixels, channel_id)
            image_content.update({name: channel_content})
        return image_content
    
    @staticmethod
    def __simple_parse(xml):
        # Ignore the namespace and parse xml, see: https://stackoverflow.com/questions/13412496
        it = et.iterparse(StringIO(xml))
        for _, el in it:
            if '}' in el.tag:
                el.tag = el.tag.split('}', 1)[1]
        return it.root

    @staticmethod
    def get_channel_content(pixels, channel):
        z_list = [plane.get('TheZ') for plane in pixels.findall('Plane') if plane.get('TheC')==channel]
        return z_list
            
    @staticmethod    
    def get_pixel_info(pixel_info):
        size = {}
        x_pixel = float(pixel_info.get('PhysicalSizeX'))
        y_pixel = float(pixel_info.get('PhysicalSizeY'))
        size.update({'pixel_size_x': x_pixel, 'pixel_size_y': y_pixel})
        x_size = int(pixel_info.get('SizeX'))
        y_size = int(pixel_info.get('SizeY'))
        c_size = int(pixel_info.get('SizeC'))
        size.update({'pixel_number_x': x_size, 'pixel_number_y': y_size, 'channel_number': c_size})
        if pixel_info.get('PhysicalSizeZ'):
            z_pixel = float(pixel_info.get('PhysicalSizeZ')) 
            size.update({'pixel_size_z': z_pixel})
        if pixel_info.get('SizeZ'):
            z_size = int(pixel_info.get('SizeZ'))
            size.update({'pixel_number_z': z_size})
        return size
    
class Image():
    def __init__(self, pkl_file, meta_file):
        self.data = read_pkl(pkl_file)
        if self.data.shape[-2:] == (1, 1):
            self.structure = 'x-y-z'
        elif self.data.shape[-1] == 1:
            self.structure = 'x-y-z-t'
        elif self.data.shape[-2] == 1:
            self.structure = 'x-y-z-c'
        self.data = np.squeeze(self.data)
        mf_surfix = meta_file.split('.')[-1] 
        if mf_surfix == 'pkl':
            self.metadata = read_meta(meta_file)
        elif mf_surfix == 'xml':
            self.metadata = Metadata(read_xml(meta_file))
        else: 
            raise TypeError("Unable to read %s file" % mf_surfix)

    def __repr__(self):
        content = 'Structure:\t{}'.format(self.structure)
        content += '\nShape:\t{}'.format(self.data.shape)
        return content

def time_lapse(basic_filename, time_max):
    for i in range(time_max):
        fn = basic_filename + '-t{}.pkl'.format(i)
        mn = basic_filename + '-t{}-meta.pkl'.format(i)
        yield Image(fn, mn)
