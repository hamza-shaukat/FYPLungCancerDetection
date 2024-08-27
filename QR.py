# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:56:48 2023

@author: Hamza
"""

import qrcode
from qrcode.constants import ERROR_CORRECT_L
from PIL import Image

data = "WIFI:S:Umer-4G;T:WPA;P:53984c7c;;"
qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data(data)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("C:/Users/Hamza/Desktop/555/wifi_qr_code.png")