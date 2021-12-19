"""
bmp.py - Pure Python bitmap Encoder/Decoder with simple functionality

The `bitmap` module can read and write Bitmap images

Current image processing:
    -convert to grayscale
    -center crop

Refer to https://en.wikipedia.org/wiki/BMP_file_format for information on bitmap file format
"""
import io
import os
import math
import numpy as np
from pprint import pprint
from struct import pack, unpack

__all__ = ['BitmapReader', 'BitmapWriter', 'Bitmap', 'fromarray']

DEFAULT_GRAYSCALE_COLORTABLE = b'\x00\x00\x00\x00\x01\x01\x01\x00\x02\x02\x02\x00\x03\x03\x03\x00\x04\x04\x04\x00\x05\x05\x05\x00\x06\x06\x06\x00\x07\x07\x07\x00\x08\x08\x08\x00\t\t\t\x00\n\n\n\x00\x0b\x0b\x0b\x00\x0c\x0c\x0c\x00\r\r\r\x00\x0e\x0e\x0e\x00\x0f\x0f\x0f\x00\x10\x10\x10\x00\x11\x11\x11\x00\x12\x12\x12\x00\x13\x13\x13\x00\x14\x14\x14\x00\x15\x15\x15\x00\x16\x16\x16\x00\x17\x17\x17\x00\x18\x18\x18\x00\x19\x19\x19\x00\x1a\x1a\x1a\x00\x1b\x1b\x1b\x00\x1c\x1c\x1c\x00\x1d\x1d\x1d\x00\x1e\x1e\x1e\x00\x1f\x1f\x1f\x00   \x00!!!\x00"""\x00###\x00$$$\x00%%%\x00&&&\x00\'\'\'\x00(((\x00)))\x00***\x00+++\x00,,,\x00---\x00...\x00///\x00000\x00111\x00222\x00333\x00444\x00555\x00666\x00777\x00888\x00999\x00:::\x00;;;\x00<<<\x00===\x00>>>\x00???\x00@@@\x00AAA\x00BBB\x00CCC\x00DDD\x00EEE\x00FFF\x00GGG\x00HHH\x00III\x00JJJ\x00KKK\x00LLL\x00MMM\x00NNN\x00OOO\x00PPP\x00QQQ\x00RRR\x00SSS\x00TTT\x00UUU\x00VVV\x00WWW\x00XXX\x00YYY\x00ZZZ\x00[[[\x00\\\\\\\x00]]]\x00^^^\x00___\x00```\x00aaa\x00bbb\x00ccc\x00ddd\x00eee\x00fff\x00ggg\x00hhh\x00iii\x00jjj\x00kkk\x00lll\x00mmm\x00nnn\x00ooo\x00ppp\x00qqq\x00rrr\x00sss\x00ttt\x00uuu\x00vvv\x00www\x00xxx\x00yyy\x00zzz\x00{{{\x00|||\x00}}}\x00~~~\x00\x7f\x7f\x7f\x00\x80\x80\x80\x00\x81\x81\x81\x00\x82\x82\x82\x00\x83\x83\x83\x00\x84\x84\x84\x00\x85\x85\x85\x00\x86\x86\x86\x00\x87\x87\x87\x00\x88\x88\x88\x00\x89\x89\x89\x00\x8a\x8a\x8a\x00\x8b\x8b\x8b\x00\x8c\x8c\x8c\x00\x8d\x8d\x8d\x00\x8e\x8e\x8e\x00\x8f\x8f\x8f\x00\x90\x90\x90\x00\x91\x91\x91\x00\x92\x92\x92\x00\x93\x93\x93\x00\x94\x94\x94\x00\x95\x95\x95\x00\x96\x96\x96\x00\x97\x97\x97\x00\x98\x98\x98\x00\x99\x99\x99\x00\x9a\x9a\x9a\x00\x9b\x9b\x9b\x00\x9c\x9c\x9c\x00\x9d\x9d\x9d\x00\x9e\x9e\x9e\x00\x9f\x9f\x9f\x00\xa0\xa0\xa0\x00\xa1\xa1\xa1\x00\xa2\xa2\xa2\x00\xa3\xa3\xa3\x00\xa4\xa4\xa4\x00\xa5\xa5\xa5\x00\xa6\xa6\xa6\x00\xa7\xa7\xa7\x00\xa8\xa8\xa8\x00\xa9\xa9\xa9\x00\xaa\xaa\xaa\x00\xab\xab\xab\x00\xac\xac\xac\x00\xad\xad\xad\x00\xae\xae\xae\x00\xaf\xaf\xaf\x00\xb0\xb0\xb0\x00\xb1\xb1\xb1\x00\xb2\xb2\xb2\x00\xb3\xb3\xb3\x00\xb4\xb4\xb4\x00\xb5\xb5\xb5\x00\xb6\xb6\xb6\x00\xb7\xb7\xb7\x00\xb8\xb8\xb8\x00\xb9\xb9\xb9\x00\xba\xba\xba\x00\xbb\xbb\xbb\x00\xbc\xbc\xbc\x00\xbd\xbd\xbd\x00\xbe\xbe\xbe\x00\xbf\xbf\xbf\x00\xc0\xc0\xc0\x00\xc1\xc1\xc1\x00\xc2\xc2\xc2\x00\xc3\xc3\xc3\x00\xc4\xc4\xc4\x00\xc5\xc5\xc5\x00\xc6\xc6\xc6\x00\xc7\xc7\xc7\x00\xc8\xc8\xc8\x00\xc9\xc9\xc9\x00\xca\xca\xca\x00\xcb\xcb\xcb\x00\xcc\xcc\xcc\x00\xcd\xcd\xcd\x00\xce\xce\xce\x00\xcf\xcf\xcf\x00\xd0\xd0\xd0\x00\xd1\xd1\xd1\x00\xd2\xd2\xd2\x00\xd3\xd3\xd3\x00\xd4\xd4\xd4\x00\xd5\xd5\xd5\x00\xd6\xd6\xd6\x00\xd7\xd7\xd7\x00\xd8\xd8\xd8\x00\xd9\xd9\xd9\x00\xda\xda\xda\x00\xdb\xdb\xdb\x00\xdc\xdc\xdc\x00\xdd\xdd\xdd\x00\xde\xde\xde\x00\xdf\xdf\xdf\x00\xe0\xe0\xe0\x00\xe1\xe1\xe1\x00\xe2\xe2\xe2\x00\xe3\xe3\xe3\x00\xe4\xe4\xe4\x00\xe5\xe5\xe5\x00\xe6\xe6\xe6\x00\xe7\xe7\xe7\x00\xe8\xe8\xe8\x00\xe9\xe9\xe9\x00\xea\xea\xea\x00\xeb\xeb\xeb\x00\xec\xec\xec\x00\xed\xed\xed\x00\xee\xee\xee\x00\xef\xef\xef\x00\xf0\xf0\xf0\x00\xf1\xf1\xf1\x00\xf2\xf2\xf2\x00\xf3\xf3\xf3\x00\xf4\xf4\xf4\x00\xf5\xf5\xf5\x00\xf6\xf6\xf6\x00\xf7\xf7\xf7\x00\xf8\xf8\xf8\x00\xf9\xf9\xf9\x00\xfa\xfa\xfa\x00\xfb\xfb\xfb\x00\xfc\xfc\xfc\x00\xfd\xfd\xfd\x00\xfe\xfe\xfe\x00\xff\xff\xff\x00'


class InputError(Exception):
    """
    Incorrect input into Bitmap object
    """


class Bitmap():
    """
    A container that stores the Bitmap image data and useful information.
    """
    def __init__(self, data, header):
        self.data = data
        self.header = header

    def write(self, f):
        writer = BitmapWriter(**self.header)

        if f.endswith('.bmp') == False:
            f += '.bmp'

        with open(f, 'wb') as f:
            writer.write(f, self.data)

    def __str__(self):
        string = ""
        for k, v in self.header.items():
            string+= f'{k}: {v} \n'
        return string

    def crop_center(self, size):
        """
        size: (height, width) or int of center crop
        """
        if isinstance(size, int):
            size = (size, size)

        assert size[1] % 4 == 0, "Width must be a multiple of 4" #dont feel like dealing with padding right now

        if size[0] > self.header['image_height']:
            size[0] = self.header['image_height']
        if size[1] > self.header['image_width']:
            size[1] = self.header['image_width']

        height = len(self.data)
        width = len(self.data[0])
        center = (height//2, width//2)
        center_image = self.data[center[0]-size[0]//2:center[0]+size[0]//2]
        for i, row in enumerate(center_image):
            center_image[i] = row[center[1]-size[1]//2:center[1]+size[1]//2]

        self.data = center_image
        self.header['image_height'] = size[0]
        self.header['image_width'] = size[1]
        self.header['image_size'] = compute_file_size(self.data)
        del center_image

    def grayscale(self):
        #grayscale only works if bits per pixel is 24
        assert self.header['bits_per_pixel'] == 24, 'Bitmap must have depth of 24 bits'
        #keep original image
        self._data = self.data
        bw_image = []
        for row in self.data:
            bw_row = []
            for channel in row:
                avg = sum(channel)//len(channel)
                bw_row.append([avg])
            bw_image.append(bw_row)

        self.data = bw_image
        self.header['bits_per_pixel'] = 8
        del bw_image


# **** helper functions *****

def calculate_row_size(bits_per_pixel, image_width):
    return math.ceil((bits_per_pixel * image_width)/32)*4

def calculate_row_padding(bits_per_pixel, image_width):
    return ((bits_per_pixel//8) * image_width) % 4

def calculate_pixel_array_size(row_size, image_height):
    return row_size * abs(image_height)

def calculate_number_of_rows(row_size, pixel_array_size):
    return pixel_array_size//row_size

def chunk(data, n):
    for i in range(0, len(data), n):
        yield i, data[i:i+n]

def make_image_array(raw_image, row_size, channels):
    image_array = []
    for i, data in chunk(raw_image, row_size):
        row = []
        for i, rgb in chunk(data, channels):
            depth = [val for val in rgb]
            row.append(depth)
        image_array.append(row)
    return image_array

def compute_file_size(image):
    num_bytes = 0
    for im in image:
        for row in im:
            num_bytes += len(row)
    return num_bytes


class BitmapReader():
    def __init__(self, file):
        if isinstance(file, bytes):
            self.file = io.BytesIO(file)
        elif isinstance(file, str):
            self.file = open(file, 'rb')
        elif isinstance(file, _io.BufferedReader):
            self.file = file
        else:
            raise InputError('Expecting a filename, bytes object, or file object')

    def decode_header(self):
        self.file.seek(0)
        file_type = self.file.read(2)
        meta = self.file.read(12)
        file_size, _, _, offset = unpack("<I2HI", meta)
        image_header = self.file.read(40)
        header_size, image_width, image_height, \
        color_planes, bits_per_pixel, compression, \
        image_size, horizontal_reso, vertical_reso,\
        color_palette, important_colors = unpack("<3I2h2I4I", image_header)

        header = {}
        header['file_size'] = file_size
        header['x'] = 0
        header['y'] = 0
        header['offset'] = offset
        header['header_size'] = header_size
        header['image_width'] = image_width
        header['image_height'] = image_height
        header['color_planes'] = color_planes
        header['bits_per_pixel'] = bits_per_pixel
        header['compression'] = compression
        header['image_size'] = image_size
        header['horizontal_resolution'] = horizontal_reso
        header['vertical_resolution'] = vertical_reso
        header['color_palette'] = color_palette
        header['important_colors'] = important_colors
        self.header = header

        row_size = calculate_row_size(bits_per_pixel, image_width)
        pixel_array_size = calculate_pixel_array_size(row_size, image_height)
        number_of_rows = calculate_number_of_rows(row_size, pixel_array_size)
        row_pad = calculate_row_padding(bits_per_pixel, image_width)

        self.meta_data = {
                    'row_size': row_size,
                    'pixel_array_size': pixel_array_size,
                    'number_of_rows': number_of_rows,
                    'row_padding': row_pad
                            }

    def open(self):
        self.decode_header()
        self.file.seek(self.header['offset'])
        raw_image = self.file.read()
        if self.header['bits_per_pixel'] >= 8:
            channels = self.header['bits_per_pixel'] // 8
        else:
            channels = self.header['bits_per_pixel']
        image_array = make_image_array(raw_image, self.meta_data['row_size'], channels)
        return Bitmap(image_array, self.header)


class BitmapWriter:
    """
    Encodes a bitmap image in pure Python
    """
    def __init__(self,
            image_width=None,
            image_height=None,
            color_planes=1,
            bits_per_pixel=24,
            compression=0,
            image_size=None,
            horizontal_resolution=0,
            vertical_resolution=0,
            color_palette=None,
            important_colors=0,
            color_table=None,
            **kwargs
            ):
        """
        Arguments:

        width, height (int): size of image in pixels
        color_planes (int): number of color planes
        bits_per_pixel (int): number of bits per pixel or color depth.
                              Typical values 1,2,4,8,16,24,32
        compression (int): compression method being used
        image_size (int): the size of the raw image in bytes
        horizontal_resolution (int): horizontal resolution of the image in
                                    pixels per meter
        vertical_resolution (int): vertical resolution of the image in
                                   pixels per meter
        color_palette (int): the number of colors being used
        important_colors (int): number of signficant colors in the image
        color_table (bytes): color table is required if bits_per_pixel <= 8.
                            The number of entries are 2^n (n is number of colors)

        """
        self.width = image_width
        self.height = image_height
        self.color_planes = color_planes
        self.bits_per_pixel = bits_per_pixel
        self.compression = compression
        self.image_size = image_size
        self.horizontal_resolution = horizontal_resolution
        self.vertical_resolution = vertical_resolution
        self.color_palette = color_palette
        self.important_colors = important_colors

        if color_table is None and bits_per_pixel <= 8:
            self.color_table = DEFAULT_GRAYSCALE_COLORTABLE
        elif color_table is not None:
            self.color_table = color_table
        else:
            self.color_table = None

    def get_offset(self):
        #TODO: modify when accept differnt types of headers
        offset = 14+40
        if self.color_table is not None:
            offset += len(self.color_table)
        return offset

    def get_file_size(self, image_array):
        num_bytes = 0
        for im in image_array:
            for row in im:
                num_bytes += len(row)
        return num_bytes

    def get_header_data(self, data):
        #TODO: make this an ordered dict at some point
        info = {
            'file_size': self.get_file_size(data) + self.get_offset(),
            'reserved_a': 0,
            'reserved_b': 0,
            'offset': self.get_offset(),
            'header_size': 40
            }
        return info

    def write_header(self, f, data):
        info = self.get_header_data(data)
        header = b''
        header += b'BM' #Only file type accepted
        header += pack('<I', info['file_size'] )
        header += pack('H', info['reserved_a'])
        header += pack('H', info['reserved_b'])
        header += pack('<I', info['offset'] )
        header += pack('<I', info['header_size'])
        header += pack('<I', self.width)
        header += pack('<I', self.height)
        header += pack('h', self.color_planes)
        header += pack('h', self.bits_per_pixel)
        header += pack('<I', self.compression)
        header += pack('<I', info['file_size'] - info['offset'])
        header += pack('<I', self.horizontal_resolution)
        header += pack('<I', self.vertical_resolution)
        header += pack('<I', self.color_palette)
        header += pack('<I', self.important_colors)

        if self.color_table is not None:
            header += self.color_table
        f.write(header)

    def write(self, f, data):
        '''
        Writes a Bitmap image to a filelike object ``f``
        '''
        #TODO: This is very inefficiant
        self.write_header(f, data)
        for row in data:
            for depth in row:
                for val in depth:
                    f.write(val.to_bytes(1, 'little'))

def fromarray(array):
    '''
    You can create a bitmap image from a 2D array

    array:
        This array can be a ```numpy`` array or a list like sequence.
        The array width must be a multiple of 4.
    '''

    if isinstance(array, np.ndarray):
        assert array.ndim == 2, 'ndarray must be 2D'
        assert len(array[1]) % 4 == 0, 'Width must be a multiple of 4'
    elif isinstance(array, list):
        try:
            islist = type(array[0]) == list
        except IndexError:
            print('Array must be 2D')
        assert islist, 'Array must be 2D'
        assert len(array[0]) % 4 == 0, 'Width must be a multiple of 4'
    else:
        raise Exception('Array must be type list or numpy.array')
