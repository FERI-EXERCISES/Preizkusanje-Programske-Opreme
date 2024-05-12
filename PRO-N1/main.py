import fnmatch
import unittest
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import os


# function 1 (GPT)
def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1


class TestBinarySearch(unittest.TestCase):
    def test_find_middle_element(self):
        result = binary_search([1, 2, 3, 4, 5], 3)
        self.assertEqual(result, 2, "Should find element at index 2")

    def test_element_not_found(self):
        result = binary_search([1, 2, 3, 4, 5], 6)
        self.assertEqual(result, -1, "Should return -1 for not found")

    def test_large_list_performance(self):
        large_list = list(range(1000000))  # seznam z 1.000.000 elementi
        target = 999999  # element na koncu seznama
        result = binary_search(large_list, target)
        self.assertEqual(result, target, "Should find element at the end of a large list")


# function 2
def convert_temperature(temp, to_scale):
    if to_scale.lower() == 'fahrenheit':
        return temp * 9 / 5 + 32
    elif to_scale.lower() == 'celsius':
        return (temp - 32) * 5 / 9
    else:
        raise ValueError("Unsupported scale")


class TestTemperatureConversion(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        result = convert_temperature(100, 'fahrenheit')
        self.assertEqual(result, 212, "Should convert 100C to 212F")

    def test_fahrenheit_to_celsius(self):
        result = convert_temperature(212, 'celsius')
        self.assertEqual(result, 100, "Should convert 212F to 100C")

    def test_negative_temperature(self):
        result = convert_temperature(-15, 'fahrenheit')
        self.assertEqual(result, 5, "Should convert -15C to 5F")

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, 'kelvin')


# function 3 (GPT)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


class TestBubbleSort(unittest.TestCase):
    def test_sorting(self):
        result = bubble_sort([3, 2, 1, 5, 4])
        self.assertEqual(result, [1, 2, 3, 4, 5], "Should sort the list")

    def test_already_sorted(self):
        result = bubble_sort([1, 2, 3, 4, 5])
        self.assertEqual(result, [1, 2, 3, 4, 5], "Should work with already sorted list")

    def test_empty_list(self):
        result = bubble_sort([])
        self.assertEqual(result, [], "Should work with empty list")

    def test_single_element_list(self):
        result = bubble_sort([1])
        self.assertEqual(result, [1], "Should work with single element list")


# function 4
def conv_2d(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    # Dimenzije
    H, W, _ = slika.shape
    N, M = jedro.shape

    # Izračun središče jedra
    C_N, C_M = N // 2, M // 2

    # Priprava vhodne slike
    tmp = np.pad(slika, ((C_N, C_M), (C_N, C_M), (0, 0)), mode='constant')

    # Pripravljenje izhodne slike
    izhod = np.zeros((H, W, 3), dtype=np.float32)

    # Izvedi konvolucijo
    for i in range(H):
        for j in range(W):
            izhod[i, j] = (tmp[i:i + N, j:j + M] * jedro).sum(axis=(0, 1))

    return izhod


class TestConv2d(unittest.TestCase):
    def TestConv2d(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        result = conv_2d(slika, jedro)
        self.assertEqual(result, [[[-6, 0, 6], [-6, 0, 6], [-6, 0, 6]]], "Should perform 2d convolution")

    def test_invalid_image_size(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        with self.assertRaises(ValueError):
            conv_2d(slika, jedro)

    def test_invalid_kernel_size(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1]], dtype=np.float32)
        with self.assertRaises(ValueError):
            conv_2d(slika, jedro)


# function 5
def open(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(slika, jedro)
    opened = binary_dilation(eroded, jedro)
    return opened.astype(bool)


class TestOpen(unittest.TestCase):
    def test_output_bool(self):
        slika = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        jedro = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        result = open(slika, jedro)
        self.assertEqual(result.dtype, bool, "Should return a boolean array")

    def test_input_not_bool(self):
        slika = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        jedro = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            open(slika, jedro)

    def test_invalid_input(self):
        with self.assertRaises(RuntimeError):
            open(None, None)


# function 6
def fft2_resample(x: np.ndarray, new_shape: tuple) -> np.ndarray:
    # Izračunamo 2D FFT in izvedemo premik frekvence 0 v središče
    fft = np.fft.fftshift(np.fft.fft2(x))
    fft_resampled = np.zeros(new_shape, dtype=complex)

    # Izračunamo minimalne dimenzije za obrezovanje ali dodajanje frekvenc
    min_x = min(x.shape[0], new_shape[0]) // 2
    min_y = min(x.shape[1], new_shape[1]) // 2

    # Obrezovanje ali dodajanje frekvenc, da dosežemo želeno velikost
    fft_resampled[new_shape[0] // 2 - min_x: new_shape[0] // 2 + min_x,
    new_shape[1] // 2 - min_y: new_shape[1] // 2 + min_y] = \
        fft[x.shape[0] // 2 - min_x: x.shape[0] // 2 + min_x, x.shape[1] // 2 - min_y: x.shape[1] // 2 + min_y]

    return np.fft.ifft2(np.fft.ifftshift(fft_resampled))


def prevzorci_sliko_fft(slika: np.ndarray, nova_visina: int, nova_sirina: int) -> np.ndarray:
    # Preveri, ali je slika barvna ali sivinska
    if slika.ndim == 3:
        # Izvajamo vzorčenje Fouriera za vsak barvni kanal posebej in shranjujemo rezultate v ustrezni kanal rezultatne slike
        rezultat = np.empty((nova_visina, nova_sirina, 3), dtype=slika.dtype)

        for i in range(3):
            rezultat[..., i] = np.abs(fft2_resample(slika[..., i], (nova_visina, nova_sirina)))
    else:
        # Če je slika enobarvna izvajamo vzorčenje Fouriera neposredno na sliki
        rezultat = np.abs(fft2_resample(slika, (nova_visina, nova_sirina)))

    return rezultat


class TestPrevzorciSlikoFFT(unittest.TestCase):
    def test_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        nova_visina, nova_sirina = 200, 200
        result = prevzorci_sliko_fft(slika, nova_visina, nova_sirina)
        self.assertEqual(result.shape, (nova_visina, nova_sirina, 3), "Should return an image with the correct shape")

    def test_invalid_input(self):
        with self.assertRaises(AttributeError):
            prevzorci_sliko_fft(None, 100, 100)

    def test_invalid_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        with self.assertRaises(ValueError):
            prevzorci_sliko_fft(slika, 0, 0)


# function 7
def RGB_v_YCbCr(slika: np.ndarray) -> np.ndarray:
    # Definiraj koeficiente za pretvorbo v YCbCr
    YCbCr_from_RGB = np.array([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]])

    YCbCr = np.dot(slika, YCbCr_from_RGB.T)
    YCbCr[:, :, [1, 2]] += 128
    return np.uint8(YCbCr)


class TestRGB_v_YCbCr(unittest.TestCase):
    def test_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        result = RGB_v_YCbCr(slika)
        self.assertEqual(result.shape, (100, 100, 3), "Should return an image with the correct shape")

    def test_invalid_input(self):
        slika = np.random.rand(100, 100)
        with self.assertRaises(ValueError):
            RGB_v_YCbCr(slika)

    def test_nonstandard_input_datatype(self):
        slika = np.random.rand(100, 100, 3).astype(np.float32)
        result = RGB_v_YCbCr(slika)
        self.assertEqual(result.dtype, np.uint8, "Should return an image of type uint8")
        self.assertEqual(result.shape, (100, 100, 3), "Should return an image with the correct shape")


# function 8
def konvolucija_fft(signal: np.ndarray, impulz: np.ndarray, rob: str) -> np.ndarray:
    N = len(signal)
    K = len(impulz)

    impulz_padded = np.pad(impulz, (0, N - K), mode='constant', constant_values=0)

    # Padding the signal and impulse response according to the specified edge handling
    if rob == 'ničle':
        signal_padded = np.pad(signal, (0, K - 1), mode='constant', constant_values=0)
        impulz_padded = np.pad(impulz, (0, N - 1), mode='constant', constant_values=0)
    elif rob == 'zrcaljen':
        signal_padded = np.pad(signal, (K - 1, K - 1), mode='reflect')
        impulz_padded = np.pad(impulz, (N - 1, N - 1), mode='reflect')
    elif rob == 'krožni':
        signal_padded = np.pad(signal, (K - 1, K - 1), mode='wrap')
        impulz_padded = np.pad(impulz, (N - 1, N - 1), mode='wrap')

    # Calculating the convolution in frequency domain
    signal_fft = np.fft.fft(signal_padded)
    impulz_fft = np.fft.fft(impulz_padded)
    output_fft = signal_fft * impulz_fft
    output = np.fft.ifft(output_fft)[:N].real

    # Reshaping the output to match the dimensions of the input signal
    if signal.ndim > 1:
        output = output.reshape((N, signal.shape[1]))

    return output


class TestKonvolucijaFFT(unittest.TestCase):
    def test_output_shape(self):
        signal = np.random.rand(100)
        impulz = np.array([0.5, 1, 0.5])
        rob = 'ničle'
        result = konvolucija_fft(signal, impulz, rob)
        self.assertEqual(result.shape, (100), "Should return an image with the correct shape")

    def test_invalid_input(self):
        signal = np.random.rand(100, 100)
        impulz = np.random.rand(5, 5)
        rob = 'ničle'
        with self.assertRaises(ValueError):
            konvolucija_fft(signal, impulz, rob)

    def test_invalid_edge_handling(self):
        signal = np.random.rand(100, 100)
        impulz = np.random.rand(5)
        rob = 'invalid'
        with self.assertRaises(UnboundLocalError):
            konvolucija_fft(signal, impulz, rob)

# function 9 (GPT)
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

class TestIsPrime(unittest.TestCase):
    def test_prime_number(self):
        result = is_prime(7)
        self.assertTrue(result, "Should return True for prime number")

    def test_non_prime_number(self):
        result = is_prime(9)
        self.assertFalse(result, "Should return False for non-prime number")

    def test_negative_number(self):
        result = is_prime(-7)
        self.assertFalse(result, "Should return False for negative number")

    def test_zero(self):
        result = is_prime(0)
        self.assertFalse(result, "Should return False for zero")

# function 10
def find_music_files(start_path, extensions):
    music_files = []

    for root, dirs, files in os.walk(start_path):
        for extension in extensions:
            for filename in fnmatch.filter(files, f'*.{extension}'):
                music_files.append(os.path.join(root, filename))
                artistname = filename.split(" - ")[0]
                songname = os.path.splitext("".join(filename.split(" - ")[1:]))[0]
                path = os.path.join(root, filename)
                filesize = os.path.getsize(os.path.join(root, filename))
                fileextension = filename.split(".")[-1]

    return music_files

class TestFindMusicFiles(unittest.TestCase):
    def test_output_type(self):
        result = find_music_files('C:/Users/simon/Music/Test', ['mp3', 'flac', 'wav'])
        self.assertIsInstance(result, list, "Should return a list")

    def test_invalid_start_path(self):
        with self.assertRaises(ValueError):
            find_music_files('invalid_path', ['mp3', 'flac', 'wav'])

    def test_invalid_extensions(self):
        with self.assertRaises(ValueError):
            find_music_files('C:/Users/Anja/PycharmProjects/PythonProject', ['invalid'])

if __name__ == '__main__':
    print("Hello, World")
    unittest.main()
