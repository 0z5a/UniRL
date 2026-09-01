import os
from hashlib import md5
from io import BytesIO
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


def evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    """与 openssl enc 的 EVP_BytesToKey 算法兼容（MD5 版）"""
    dtot = b''
    prev = b''
    while len(dtot) < (key_len + iv_len):
        prev = md5(prev + password + salt).digest()
        dtot += prev
    key = dtot[:key_len]
    iv = dtot[key_len:key_len+iv_len]
    return key, iv


def aes_encrypt(plaintext: bytes, password: str, save_file: str | Path = None) -> BytesIO | str | Path:
    salt = os.urandom(8)

    key_len, iv_len = 16, 16  # AES-128-CBC

    # 派生密钥和IV
    key, iv = evp_bytes_to_key(password.encode('utf-8'), salt, key_len, iv_len)

    cipher = AES.new(key, AES.MODE_CBC, iv)

    # 对明文进行PKCS7填充并加密
    padded_plaintext = pad(plaintext, AES.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)

    # 构建OpenSSL格式的加密数据：Salted__ + salt + ciphertext
    enc_data = b'Salted__' + salt + ciphertext

    if save_file is not None:
        with open(save_file, 'wb') as f:
            f.write(enc_data)

    return BytesIO(enc_data)


def aes_decrypt(enc_file: str, password: str, save_file=None) -> BytesIO | str | Path:
    with open(enc_file, 'rb') as f:
        enc_data = f.read()

    # OpenSSL 格式：前8字节是 "Salted__"，后8字节是 salt
    assert enc_data[:8] == b'Salted__', "Invalid encrypted file."
    salt = enc_data[8:16]
    ciphertext = enc_data[16:]

    key_len, iv_len = 16, 16  # AES-128-CBC
    # 派生密钥和IV
    key, iv = evp_bytes_to_key(password.encode('utf-8'), salt, key_len, iv_len)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)

    # 去除 PKCS7 填充
    pad_len = decrypted[-1]
    decrypted_data = decrypted[:-pad_len]

    if save_file is not None:
        with open(save_file, 'wb') as f:
            f.write(decrypted_data)
        return save_file

    return BytesIO(decrypted_data)
