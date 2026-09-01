import base64
import hashlib
import io
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm import tqdm


def download_data(url, save_to=None, headers=None, proxies=None, ret_binary=False,
                  timeout=5) -> bytes | Path:
    """
    Download data from an url. Using proxy is allowed.

    Parameters
    ----------
    url: str
        A direct url to download the assets.
    save_to: str, pathlib.Path
        If it is None, `ret_binary` must be set as True.
        If it has a suffix, we refer it as a file path.
        If it doesn't have a suffix, we detect its binary type and save it as a file.
    headers: dict
        The headers used for requests.
    proxies: dict
        If True, use the default predefined proxies in 'image_pipeline.constants'.
        If a dict, using the given proxies. The dict must be a format of {'<proxy>': '<url>'}.
        For example:
        {
            'http_proxy': 'http://xxx.com:8081',
            'https_proxy': 'http://xxx.com:8081',
        }
    ret_binary: bool
        If True, return binary data, else return saving path.
    timeout: float
        The timeout for requests.

    Returns
    -------
    bytes or Path
        If `ret_binary` is True, a bytes buffer will be returned.
        Else, a pathlib.Path object indicating the saving path will be returned.
    """
    if headers is None:
        headers = {}
    if not proxies:
        r = requests.get(url, timeout=timeout)
    else:
        r = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)

    if save_to is None:
        if not ret_binary:
            raise ValueError(f"If `save_to` is None, `ret_binary` must be set as True.")
    else:
        save_to = Path(save_to)
        if save_to.suffix == "":
            img_type = binary_image_type(r.content[:32])
            save_to = save_to.with_suffix("." + img_type)
        save_to.parent.mkdir(parents=True, exist_ok=True)

        with save_to.open("wb") as f:
            f.write(r.content)

    if ret_binary:
        return r.content

    return save_to


def binary_image_type(data):
    """
    检测图片的类型
    :param data: 图片二进制
    :return:
    """
    data = data[:32]

    if data[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'

    elif data.startswith(b'\211PNG\r\n\032\n'):
        return 'png'

    elif data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

    elif data[:2] in (b'MM', b'II'):
        return 'tiff'

    elif data.startswith(b'\001\332'):
        return 'rgb'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'14' and data[2] in b' \t\n\r':
        return 'pbm'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'25' and data[2] in b' \t\n\r':
        return 'pgm'

    elif len(data) >= 3 and data[0] == ord(b'P') and data[1] in b'36' and data[2] in b' \t\n\r':
        return 'ppm'

    elif data.startswith(b'\x59\xA6\x6A\x95'):
        return 'rast'

    elif data.startswith(b'#define '):
        return 'xbm'

    elif data.startswith(b'BM'):
        return 'bmp'

    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'webp'

    elif data.startswith(b'\x76\x2f\x31\x01'):
        return 'exr'

    else:
        return None


# ==========================================
#  Image reader
# ==========================================

def read_local_image(src, convert_mode="RGB", apply_exif=False):
    with Image.open(src) as image:
        image.load()
    if apply_exif:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(f"Warning: Failed to apply exif transpose for image {src}: {e}")
    if convert_mode is not None:
        image = image.convert(convert_mode)
    return image


def read_binary_image(binary, convert_mode="RGB", apply_exif=False):
    image_bytes = io.BytesIO(binary)
    image_bytes.seek(0)
    return read_local_image(image_bytes, convert_mode, apply_exif)


def read_base64_image(base64_str: str, convert_mode="RGB", apply_exif=False) -> Image.Image:
    binary = base64.b64decode(base64_str)
    return read_binary_image(binary, convert_mode, apply_exif)


def read_url_image(url, proxies=None, timeout=5, convert_mode="RGB", apply_exif=False):
    binary = download_data(url, proxies=proxies, ret_binary=True, timeout=timeout)
    return read_binary_image(binary, convert_mode, apply_exif=apply_exif)


IMAGE_READER = dict(
    local_path=read_local_image,
    base64=read_base64_image,
    url=read_url_image,
)


def read_image(reader_type, src, max_retry=3, logger=None):
    image = None
    for _ in range(max_retry):
        try:
            image = IMAGE_READER[reader_type](src)
            break
        except Exception as e:
            if logger is not None:
                logger.warning(f"Failed to read image from {src}: {e}")
    return image


# Deprecated. Just for bc
def get_image(src, proxies=True):
    src = str(src)
    if src.startswith('http'):
        return read_url_image(src, proxies=proxies)
    elif src.startswith('images/'):
        src = Path('vis') / src
        return Image.open(src).convert("RGB")
    else:
        return Image.open(src).convert("RGB")


# ==========================================
#  Image writer
# ==========================================

def image_to_binary(image: Image.Image, fmt="PNG"):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=fmt)
    image_bytes.seek(0)
    return image_bytes.read()


def image_to_base64(image: Image.Image, fmt="PNG") -> str:
    return base64.b64encode(image_to_binary(image, fmt)).decode()


def binary_to_base64(binary: bytes) -> str:
    return base64.b64encode(binary).decode()


def url_to_base64(url, proxies=None, timeout=5):
    binary = download_data(url, proxies=proxies, ret_binary=True, timeout=timeout)
    return binary_to_base64(binary)


def url_to_base64_and_mime(url, proxies=None, timeout=5):
    binary = download_data(url, proxies=proxies, ret_binary=True, timeout=timeout)
    return binary_to_base64(binary), binary_image_type(binary)

def local_path_to_base64_and_mime(local_path):
    binary = Path(local_path).read_bytes()
    return binary_to_base64(binary), binary_image_type(binary)

def save_url_to_image(args):
    url, save_path, fmt = args
    if Path(save_path).exists():
        return

    try:
        image = read_url_image(url, apply_exif=True)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path, format=fmt)
    except Exception as e:
        print(url, e)


def p_save_urls(data, num=32):
    with Pool(processes=num) as pool:
        list(tqdm(
            pool.imap(
                save_url_to_image, data
            ),
            total=len(data)
        ))


# ==========================================
#  Image metrics
# ==========================================

def contrast_metric(image):
    """
    Calculate the contrast metric of an image.

    Parameters
    ----------
    image: Image.Image
        PIL Image

    Returns
    -------
    float
        Contrast metric

    """
    image = np.array(image.convert('L'))
    std = image.std()
    return std


def saturation_metric(image):
    """
    Calculate the saturation metric of an image.

    Parameters
    ----------
    image: Image.Image
        PIL Image

    Returns
    -------
    float
        Saturation metric

    """
    image = np.array(image.convert('HSV'))
    saturation = image[:, :, 1]
    mean = saturation.mean()
    return mean


def md5sum(*, file=None, binary=None, url=None, proxies=None):
    if file is not None:
        with open(file, 'rb') as f:
            content = f.read()
    elif binary is not None:
        content = binary
    elif url is not None:
        content = download_data(url, proxies=proxies, ret_binary=True)
    else:
        raise ValueError(f"At least one source is needed.")

    md5hash = hashlib.md5(content)
    md5 = md5hash.hexdigest()
    return md5


def md5sum_binary(x):
    return md5sum(binary=x)


def md5sum_file(x):
    return md5sum(file=x)


def md5sum_url(x):
    return md5sum(url=x)


# ==========================================
#  Image ops
# ==========================================

def resize_image(image, size):
    """ Resize the short side of an image to the given size. """
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(size / width * height)
    else:
        new_height = size
        new_width = int(size / height * width)
    return image.resize((new_width, new_height))

def unpad_image(pil_image, ori_width, ori_height):
    """ Unpad the image to the original size. """
    tw, th = pil_image.size
    w = ori_width
    h = ori_height
    tr = th / tw
    r = h / w

    # resize
    if r < tr:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    else:
        resize_height = th
        resize_width = int(round(th / h * w))

    # pad
    pad_width = tw - resize_width
    pad_height = th - resize_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pil_image = pil_image.crop((pad_left, pad_top, pad_left + resize_width, pad_top + resize_height))
    return pil_image