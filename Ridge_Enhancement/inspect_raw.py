import sys 
sys.path.append('.')
import Image
from skimage import exposure, restoration, util

params = Image.Params(background_radius=20)

stack = Image.load_stack('tiff/ch20_URA7_URA8_001-crop1.tif')
frame = stack[50]

img = util.img_as_float32(frame)
print(f"Float frame min/max/mean: {img.min():.5f}, {img.max():.5f}, {img.mean():.5f}")

img_rescaled = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
print(f"Rescaled frame min/max/mean: {img_rescaled.min():.5f}, {img_rescaled.max():.5f}, {img_rescaled.mean():.5f}")

bg = restoration.rolling_ball(img_rescaled, radius=params.background_radius)
print(f"Bg min/max/mean: {bg.min():.5f}, {bg.max():.5f}, {bg.mean():.5f}")

sub_img = img_rescaled - bg
print(f"Subtracted img min/max: {sub_img.min():.5f}, {sub_img.max():.5f}")

clipped = sub_img.clip(0, None)
final = exposure.rescale_intensity(clipped, out_range=(0.0, 1.0))
print(f"Final img min/max/mean: {final.min():.5f}, {final.max():.5f}, {final.mean():.5f}")
