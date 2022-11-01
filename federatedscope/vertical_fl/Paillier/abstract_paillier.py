# You can refer to pyphe for the detail implementation. (
# https://github.com/data61/python-paillier/blob/master/phe/paillier.py)
# Or implement an effective version of Paillier (<Public-key cryptosystems
# based on composite degree residuosity classes>)

DEFAULT_KEYSIZE = 3072


def generate_paillier_keypair(n_length=DEFAULT_KEYSIZE):
    """Generate public key and private key used Paillier`.

    Args:
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    n = p = q = None
    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    return public_key, private_key


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.
    """
    def __init__(self, n):
        """
        Initialize the class with the given number.

        Args:
            self: write your description
            n: write your description
        """
        pass

    def encrypt(self, value):
        """
        Encrypt the value.

        Args:
            self: write your description
            value: write your description
        """
        # We only provide an abstract implementation here

        return value


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.
    """
    def __init__(self, public_key, p, q):
        """
        This is the primary public method for RSA.

        Args:
            self: write your description
            public_key: write your description
            p: write your description
            q: write your description
        """
        pass

    def decrypt(self, encrypted_number):
        """
        Decrypts the encrypted number.

        Args:
            self: write your description
            encrypted_number: write your description
        """
        # We only provide an abstract implementation here

        return encrypted_number
