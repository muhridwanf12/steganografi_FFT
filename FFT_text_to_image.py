import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import io


def normalizedRGB(image):
    if image.ndim == 3: 
        return image / 255.0
    elif image.ndim == 2:  
        return np.stack((image, image, image), axis=-1) / 255.0

def centralize(img, side=0.06, clip=False):
    img = img.real.astype(np.float64)
    thres = img.size * side

    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.0
        s = np.sum(img < m)
        if s < thres:
            l = m
        else:
            r = m
    low = l            

    l = img.min()
    r = img.max()
    while l + 1 <= r:
        m = (l + r) / 2.0
        s = np.sum(img > m)
        if s < thres:
            r = m
        else:
            l = m            

    high = max(low + 1, r)
    img = (img - low) / (high - low)

    if clip:
        img = np.clip(img, 0, 1)

    return img, low, high

def shuffleGen(size, secret=None):
    r = np.arange(size)
    if secret:  # Not None, ""
        np.random.seed(secret)
        np.random.shuffle(r)
    return r

def xmapGen(shape, secret=None):
    xh = shuffleGen(shape[0], secret).reshape((-1, 1))
    xw = shuffleGen(shape[1], secret)
    return xh, xw

def encodeImage(oa, ob, xmap=None, margins=(1, 1), alpha=None):
    na = normalizedRGB(oa)
    nb = normalizedRGB(ob)
    fa = np.fft.fft2(na[..., 0], axes=(0, 1))  # FFT on the red channel

    pb_shape = (fa.shape[0] // 2 - margins[0] * 2, fa.shape[1] - margins[1] * 2)
    pb = np.zeros(pb_shape)
    min_shape = (min(pb.shape[0], nb.shape[0]), min(pb.shape[1], nb.shape[1]))
    pb[:min_shape[0], :min_shape[1]] = nb[:min_shape[0], :min_shape[1], 0]

    low = 0
    if alpha is None:
        _, low, high = centralize(fa)
        alpha = (high - low)
        st.write(f"encodeImage: alpha = {alpha}")

    if xmap is None:
        xh, xw = xmapGen(pb.shape[:2])
    else:
        xh, xw = xmap[:2]

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(pb.shape)

    fa[+margins[0] + xh, +margins[1] + xw] += pb * alpha  
    fa[-margins[0] - xh, -margins[1] - xw] += pb * alpha  

    xa = np.fft.ifft2(fa, axes=(0, 1))
    xa = xa.real
    xa = np.clip(xa, 0, 1)

    na[..., 0] = xa 

    return na, fa

def encodeText(oa, text, *args, **kwargs):
    font = ImageFont.truetype("consola.ttf", oa.shape[0] // 7)
    renderSize = font.getbbox(text)[2:] 
    padding = min(renderSize) * 2 // 10
    renderSize = (renderSize[0] + padding * 2, renderSize[1] + padding * 2)
    textImg = Image.new('RGB', renderSize, (0, 0, 0))
    draw = ImageDraw.Draw(textImg)
    draw.text((padding, padding), text, (255, 255, 255), font=font)
    ob = np.asarray(textImg)
    return encodeImage(oa, ob, *args, **kwargs)

def decodeImage(xa, xmap=None, margins=(1, 1), oa=None, full=False):
    na = normalizedRGB(xa)
    fa = np.fft.fft2(na[..., 0], axes=(0, 1)) 
        
    if xmap is None:
        xh = shuffleGen(xa.shape[0] // 2 - margins[0] * 2).reshape((-1, 1))
        xw = shuffleGen(xa.shape[1] - margins[1] * 2)
    else:
        xh, xw = xmap[:2]
        
    if oa is not None:
        noa = normalizedRGB(oa)
        foa = np.fft.fft2(noa[..., 0], axes=(0, 1))
        fa -= foa
        
    if full:
        nb, _, _ = centralize(fa, clip=True)
    else:
        nb, _, _ = centralize(fa[+margins[0] + xh, +margins[1] + xw], clip=True)
        
    return nb

# Streamlit UI
st.title("Steganography using FFT")

tab1, tab2 = st.tabs(["Encode", "Decode"])

with tab1:
    st.header("Encode Image")
    uploaded_file = st.file_uploader("Choose an image to encode", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        text_input = st.text_input("Enter the text to hide in the image:")
        if st.button("Encode"):
            encoded_image_np, _ = encodeText(image_np, text_input, margins=(1, 1), alpha=None)
            encoded_image = Image.fromarray((encoded_image_np * 255).astype(np.uint8))
            st.image(encoded_image, caption='Encoded Image')
            # Store the encoded image in session state
            st.session_state['encoded_image_np'] = encoded_image_np

            # Save encoded image
            buf = io.BytesIO()
            encoded_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="Download Encoded Image", data=byte_im, file_name="encoded_image.png", mime="image/png")

with tab2:
    st.header("Decode Image")
    uploaded_file_decode = st.file_uploader("Choose an encoded image to decode", type=["jpg", "png", "jpeg"])

    if uploaded_file_decode is not None:
        encoded_image = Image.open(uploaded_file_decode)
        if encoded_image.mode != 'RGB':
            encoded_image = encoded_image.convert('RGB')
        encoded_image_np = np.array(encoded_image)

        if st.button("Decode"):
            decoded_image_np = decodeImage(encoded_image_np)
            decoded_image = Image.fromarray((decoded_image_np * 255).astype(np.uint8))
            st.image(decoded_image, caption='Decoded Image')

            # Save decoded image
            buf = io.BytesIO()
            decoded_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="Download Decoded Image", data=byte_im, file_name="decoded_image.png", mime="image/png")
