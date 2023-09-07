from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from itertools import product, zip_longest

def valid_input(image_size: tuple[int, int], tile_size: tuple[int, int], ordering: list[int]) -> bool:
    """
    Return True if the given input allows the rearrangement of the image, False otherwise.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once.

    Assumptions: 
    - tiles are always square
    - image is always square

    """

    n_reordered_tiles = len(ordering)
    expected_n_tiles = (image_size[0] / tile_size[0]) * (image_size[1] / tile_size[1])

    return (
        # total image size is divisible by number of tiles evenly
        # (image_size[0] + image_size[1]) % n_reordered_tiles == 0

        # tile size is a factor of image size
        image_size[0] % tile_size[0] == 0
        and image_size[1] % tile_size[1] == 0

        # ordering has the same number of tiles and uses each tile exactly once
        and len(set(ordering)) == n_reordered_tiles
        and expected_n_tiles == n_reordered_tiles
        )
 
 
def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.

    >>> grouper('ABCDEFG', 3, 'x')
    ['ABC', 'DEF', 'Gxx']
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

   
def crop_tiles(image, tile_size):
    """Crop the given image into tiles and return them as a list of numpy arrays."""

    width, height, tile_width, tile_height = image.size[0], image.size[1], tile_size[0], tile_size[1]

    grid = list(product(range(0, height-height%tile_height, tile_height), range(0, width-width%tile_width, tile_width)))

    cropped_imgs = []
    for i, (w, h) in enumerate(grid):
        box = (h, w, h+tile_width, w+tile_height)
        cr = image.crop(box)
        cropped_imgs.append((cr))

    return cropped_imgs

    
def merge_images_horizontal(images, mode):
    w = sum([im.size[0] for im in images])
    h = max([im.size[1] for im in images])
    x_offset, y_offset = 0, 0

    new_im = Image.new(mode, (w,h))

    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

    return new_im

def merge_images_vertical(images, mode):
    w = max([im.size[0] for im in images])
    h = sum([im.size[1] for im in images])
    x_offset, y_offset = 0, 0

    new_im = Image.new(mode, (w,h))

    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        y_offset += im.size[1]

    return new_im


def rearrange_tiles(image_path: str, tile_size: tuple[int, int], ordering: list[int], out_path: str) -> None:
    """
    Rearrange the image.

    The image is given in `image_path`. Split it into tiles of size `tile_size`, and rearrange them by `ordering`.
    The new image needs to be saved under `out_path`.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once. If these conditions do not hold, raise a ValueError with the message:
    "The tile size or ordering are not valid for the given image".
    """

    img = Image.open(image_path)

    # run checks, raise ValueError if invalid image/size/etc 
    assert img is not None, "file could not be read, check with os.path.exists()"
    if not valid_input(img.size, tile_size, ordering): 
        raise ValueError("The tile size or ordering are not valid for the given image")

    # create tiles from full scrambled image
    image_tiles = crop_tiles(img, tile_size)

    # reorder tiles according to the given ordering
    reordered_images = [np.array(image_tiles[ix]) for ix in ordering]

    # merge tiles back together, first by rows then merge the rows
    row_groupings = [list(x) for x in list(grouper(ordering, img.size[0] // tile_size[0]))]
    final_image = merge_images_vertical([merge_images_horizontal([image_tiles[ix] for ix in row_groupings[x]], mode=img.mode) 
                                           for x in range(len(row_groupings))], mode=img.mode)

    if out_path is None:
        final_image.show()
    
    if out_path is not None: 
        final_image.save(out_path)


def rearrange_tiles_ij(image_path: str, tile_size: tuple[int, int], ordering: list[int], out_path: str) -> None:
    """
    Rearrange the image.

    The image is given in `image_path`. Split it into tiles of size `tile_size`, and rearrange them by `ordering`.
    The new image needs to be saved under `out_path`.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once. If these conditions do not hold, raise a ValueError with the message:
    "The tile size or ordering are not valid for the given image".
    """

    img = Image.open(image_path)

    # run checks, raise ValueError if invalid image/size/etc 
    assert img is not None, "file could not be read, check with os.path.exists()"
    if not valid_input(img.size, tile_size, ordering): 
        raise ValueError("The tile size or ordering are not valid for the given image")

    input_img = Image.open(img_path)
    numerical_img = np.array(input_img)

    new_img = np.zeros_like(numerical_img)
    arg_order = np.argsort(ordering)

    # only actually care about one dimension of tiles for counting
    n_tiles_x = input_img.size[0] // tile_size[0]
    n_tiles_y = input_img.size[1] // tile_size[1]


    for i, ix in enumerate(ordering):  # Loop through numbers 0 to 4
        x_scrambled = int(ix // n_tiles_x)
        y_scrambled = int(ix % n_tiles_x)

        x = int(i // n_tiles_x)
        y = int(i % n_tiles_x)

        new_img[x*tile_size[0]:(x+1)*tile_size[0], y*tile_size[1]:(y+1)*tile_size[1], :] = numerical_img[x_scrambled*tile_size[0]:(x_scrambled+1)*tile_size[0], y_scrambled*tile_size[1]:(y_scrambled+1)*tile_size[1], :]

    if out_path is None:
        new_img.show()
    
    if out_path is not None: 
        new_img.save(out_path)
