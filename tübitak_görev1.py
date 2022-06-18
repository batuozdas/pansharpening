import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io
import pywt
import tifffile
from sklearn.decomposition import PCA
from skimage.exposure import match_histograms
from scipy import signal
import os,warnings
from skimage.metrics import structural_similarity

warnings.simplefilter(action='ignore', category=FutureWarning)
enterpolation_methods = ['Nearest','Bilinear','Bicubic']
sharpening_methods = ['Brovey','PCA','DWT','HSV','IHS','HPF','SFIM']

l1_path = str(input('L1 görüntü bandlarının yolunu giriniz:'))
l1r_path = str(input('L1R görüntü bandlarının yolunu giriniz:'))
enterpolation_method = str(input('Enterpolasyon yöntemi seçiniz({}):'.format(enterpolation_methods))).lower()
pansharpening_method = str(input('Pankeskinleştirme yöntemi seçiniz({}):'.format(sharpening_methods))).lower()


def reading_imgs(path):
    bands = []
    for subdir in next(os.walk(path))[1]:
        bands.append(cv2.imread(path + '/' + subdir + '/' + 'image.tif', 0))
    return bands

def aligning(img, ref_img):
    rows, cols, d = img.shape
    ref_r, ref_c, ref_d = ref_img.shape
    new_img = np.zeros(ref_img.shape)
    new_img[:, :, :] = img[: ref_r, cols - ref_c:, :]
    aligned_img = np.zeros(new_img.shape)
    for i in range(ref_c):
        aligned_img[:, i, :] = new_img[:, ref_c - i - 1, :]
    aligned_img = aligned_img.astype('uint8')
    return aligned_img


def aligning_pan(nalign_pan, ref_pan):
    cols = nalign_pan.shape[1]
    ref_r, ref_c = ref_pan.shape
    new_img = np.zeros(ref_pan.shape)
    new_img[:, :] = nalign_pan[: ref_r, cols - ref_c:]
    aligned_img = np.zeros(new_img.shape)
    for i in range(ref_c):
        aligned_img[:, i] = new_img[:, ref_c - i - 1]
    aligned_img = aligned_img.astype('uint8')
    return aligned_img


def enterpolation(img, pan, algorithm='bicubic'):
    ref_row, ref_col = pan.shape
    if algorithm == 'nearest':
        output = cv2.resize(img, dsize=(ref_col, ref_row), interpolation=cv2.INTER_NEAREST)
    elif algorithm == 'bilinear':
        output = cv2.resize(img, dsize=(ref_col, ref_row), interpolation=cv2.INTER_LINEAR)
    else:
        output = cv2.resize(img, dsize=(ref_col, ref_row), interpolation=cv2.INTER_CUBIC)
    return output


def pan_histogram_matching(img, pan):
    d = img.shape[2]
    output_img2 = np.zeros(img.shape)
    for band in range(d):
        output_img2[:, :, band] = match_histograms(pan, img[:, :, band], multichannel=False)
    output_img_hist = output_img2.astype('uint8')
    output_img_gray = cv2.cvtColor(output_img_hist, cv2.COLOR_RGB2GRAY)
    return output_img_gray


def pansharpening(img, pan_img, algorithm):
    pan_img = pan_histogram_matching(img, pan_img)
    if algorithm == 'dwt':
        level = 1
        wavelet_func = 'haar'
        rows, cols, bandnum = img.shape
        pan_dwt = pywt.wavedec2(pan_img, wavelet_func, level=level)
        panvec, pan_slices, pan_shapes = pywt.ravel_coeffs(pan_dwt)
        new_vec = np.zeros((panvec.shape[0], bandnum))
        ms_vec = np.zeros((panvec.shape[0], bandnum))
        for band in range(bandnum):
            new_vec[:, band] = panvec
            ms_dwt = pywt.wavedec2(img[:, :, band], wavelet_func, level=level)
            msvec, ms_slices, ms_shapes = pywt.ravel_coeffs(ms_dwt)
            ms_vec[:, band] = msvec

        for j in range(0, pan_shapes[0][0] * pan_shapes[0][1]):
            new_vec[j, :] = ms_vec[j, :]
        pansharpened_img = np.zeros((rows, cols, bandnum))
        for band in range(bandnum):
            new_img = pywt.unravel_coeffs(new_vec[:, band], pan_slices, pan_shapes, output_format='wavedec2')
            pansharpened_img[:, :, band] = pywt.waverec2(new_img, wavelet_func)

        pansharpened_img[pansharpened_img < 0] = 0
        pansharpened_img[pansharpened_img > 255] = 255
        return pansharpened_img.astype('uint8')

    elif algorithm == 'pca':
        rows, cols, d = img.shape
        ms_vec = img.reshape((rows * cols, d))
        pca = PCA(n_components=d)
        ms_vec_pca = pca.fit_transform(ms_vec)
        ms_arr_pca = ms_vec_pca.reshape((rows, cols, d))
        P = (pan_img - pan_img.mean()) * (np.std(ms_arr_pca) / pan_img.std()) + np.mean(ms_arr_pca)
        ms_arr_pca[:, :, 0] = P
        pansharpened_img = pca.inverse_transform(ms_arr_pca)
        pansharpened_img[pansharpened_img < 0] = 0
        pansharpened_img[pansharpened_img > 255] = 255
        return pansharpened_img.astype('uint8')

    elif algorithm == 'hsv':
        rows, cols, d = img.shape
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = pan_img
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    elif algorithm == 'brovey':
        rows, cols, d = img.shape
        I = img.sum(axis=2) / d
        pan_ms = (pan_img - I) / I
        pan_ms[np.isnan(pan_ms)] = 0
        pan_ms[np.isinf(pan_ms)] = 255
        pansharpened_img = np.zeros(img.shape)
        pan_ms[np.isnan(pan_ms)] = 0
        for band in range(d):
            pansharpened_img[:, :, band] = img[:, :, band] + pan_ms * img[:, :, band]

        pansharpened_img[np.isnan(pansharpened_img)] = 0
        pansharpened_img[np.isinf(pansharpened_img)] = 255
        return pansharpened_img.astype('uint8')

    elif algorithm == 'hpf':
        rows, cols, d = img.shape
        r = 2
        h_kernel = (-1) * np.ones((2 * r + 1, 2 * r + 1))
        h_kernel[int(h_kernel.shape[0] / 2), int(h_kernel.shape[1] / 2)] = ((2 * r + 1) ** 2) - 1
        h_kernel = h_kernel.astype('int')
        hpf_sharpened = cv2.filter2D(pan_img, -1, kernel=h_kernel)
        sharpened_w = 0.3 * hpf_sharpened
        pansharpened_img = np.zeros(img.shape)
        for band in range(d):
            pansharpened_img[:, :, band] = img[:, :, band] + sharpened_w
        return pansharpened_img.astype('uint8')

    elif algorithm == 'ıhs':
        rows, cols, d = img.shape
        I = np.sum(img, axis=2) / d
        pan_mean = np.mean(pan_img)
        I_mean = np.mean(I)
        P = pan_img * (np.std(I) / np.std(pan_img))
        P = P - pan_mean - I_mean
        P = (P - pan_mean) * np.std(I) / np.std(pan_img) + I_mean
        new_P = P - I
        pansharpened_img = np.zeros(img.shape)
        for band in range(d):
            pansharpened_img[:, :, band] = img[:, :, band] + new_P
        pansharpened_img2 = pansharpened_img + 128
        return pansharpened_img2.astype('uint8')
    elif algorithm == 'sfim':
        rows, cols, d = img.shape
        r = 2
        kernel = np.ones((r, r)) / r ** 2
        I = cv2.filter2D(pan_img, -1, kernel)
        panI = pan_img / I
        pansharpened_img = np.zeros(img.shape)
        for band in range(d):
            pansharpened_img[:, :, band] = img[:, :, band] * panI
        gama = 0.5
        pansharpened_img = pansharpened_img ** gama
        norm_img = ((pansharpened_img - np.min(pansharpened_img)) / (
                np.max(pansharpened_img) - np.min(pansharpened_img))) * 255
        return norm_img.astype('uint8')


class results:
    def __init__(self,img,ref_img):
        self.img = img
        self.ref = ref_img

    def table_results(self):
        dic = {'MSE':0, 'RMSE':0,'PSNR':0,'CC':0,'SSIM':0,'ERGAS':0,'UIQI':0}
        dic['MSE'] = self.mse(); dic['RMSE'] = self.rmse(); dic['PSNR'] = self.psnr()
        dic['CC'] = self.cc(); dic['SSIM'] = self.ssim()
        dic['ERGAS'] = self.ergas(); dic['UIQI'] = self.uıqı()
        table = pd.DataFrame(dic,index=[0])
        return table

    def mse(self):
        return np.mean((self.ref - self.img) ** 2)
    def rmse(self):
        return np.sqrt(self.mse())
    def psnr(self):
        Max = np.max(self.img)
        return 10 * np.log10(Max ** 2 / self.mse())

    def cc(self):
        mean_img = np.mean(self.img)
        mean_ref_img = np.mean(self.ref)
        img_s = self.img - mean_img
        ref_img_s = self.ref - mean_ref_img
        pay = ref_img_s * img_s
        pay_sum = np.sum(pay)
        img_s2 = img_s ** 2
        ref_img_s2 = ref_img_s ** 2
        img_s2_sum = np.sum(img_s2)
        ref_img_s2_sum = np.sum(ref_img_s2)
        payda = img_s2_sum * ref_img_s2_sum
        payda2 = np.sqrt(payda)
        return pay_sum / payda2

    def ssim(self):
        return structural_similarity(self.img,self.ref)

    def ergas(self):
        ratio = self.img.shape[0] / self.ref.shape[0]
        rmse2 = self.rmse() ** 2
        mean_ref2 = np.mean(self.ref) ** 2
        sqrt = np.sqrt(rmse2 / mean_ref2)
        return 100 * ratio * sqrt

    def uıqı(self):
        cc = self.cc()
        mean_img = np.mean(self.img)
        mean_ref = np.mean(self.ref)
        std_img = np.std(self.img)
        std_ref = np.std(self.ref)
        pay = 4 * mean_img * mean_ref * cc
        payda = ((mean_img ** 2) + (mean_ref ** 2)) * ((std_img ** 2) + (std_ref ** 2))
        return pay / payda


def plot_results(original_img, pansharpened_img, l1r_img):
    fig,((ax1,ax2,ax3)) = plt.subplots(ncols=3)
    ax1.imshow(original_img); ax1.set_title('Orijinal Görüntü')
    ax2.imshow(pansharpened_img); ax2.set_title('Pankeskinleştirilmiş Görüntü')
    ax3.imshow(l1r_img); ax3.set_title('Referans L1R Görüntüsü')


#READING IMAGES
l1_bands = reading_imgs(l1_path)
l1r_bands = reading_imgs(l1r_path)
l1_pan, l1_r, l1_g, l1_b = l1_bands
l1r_pan, l1r_r, l1r_g, l1r_b = l1r_bands
l1_img = np.dstack((l1_r, l1_g, l1_b))
l1r_img = np.dstack((l1r_r, l1r_g, l1r_b))

# ALIGNING IMAGES
aligned_ms_img = aligning(l1_img,l1r_img)
aligned_pan_img = aligning_pan(l1_pan,l1r_pan)

#ENTERPOLATING MS IMAGES
enterpolated_img = enterpolation(aligned_ms_img,aligned_pan_img,algorithm=enterpolation_method)
enterpolated_l1r = enterpolation(l1r_img,l1r_pan,algorithm=enterpolation_method)

# HISTOGRAM MATCHING
pan_hist_matched = pan_histogram_matching(enterpolated_img,aligned_pan_img)

# PANSHARPENING
pansharpened_img = pansharpening(enterpolated_img,pan_hist_matched,algorithm=pansharpening_method)

# RESULTS
red_results = results(pansharpened_img[:,:,0],enterpolated_l1r[:,:,0]).table_results()
green_results = results(pansharpened_img[:,:,1],enterpolated_l1r[:,:,1]).table_results()
blue_results = results(pansharpened_img[:,:,2],enterpolated_l1r[:,:,2]).table_results()
table_results = (red_results + green_results + blue_results) / 3
print(table_results)
plot_results(enterpolated_img,pansharpened_img,enterpolated_l1r)

