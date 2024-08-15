from PIL import Image

def save_black_and_white_image(image, path):
    # Convert to 0s and 255s
    image = image * 255
    image = Image.fromarray(image)
    image = image.convert("L")
    image.save(path)
