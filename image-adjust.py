from PIL import Image

def modify_image(input_path, output_path, size=(1024, 1024), bit_depth=24):

    original_image = Image.open(input_path)

    resized_image = original_image.resize(size, Image.ANTIALIAS)

    if original_image.mode != 'RGB':
        converted_image = resized_image.convert('RGB')
    else:
        converted_image = resized_image

    converted_image.save(output_path, bits=bit_depth)

# 用法示例
input_image_path = 'file/to/path'
output_image_path = 'file/to/path'

modify_image(input_image_path, output_image_path)
