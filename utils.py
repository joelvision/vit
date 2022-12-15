import matplotlib.pyplot as plt

def imshow(img):
    plt.figure(figsize=(100, 100))
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.savefig('sample/patch_img.png')