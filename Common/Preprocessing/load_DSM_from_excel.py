import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def temp():
    df = pd.read_excel('DSM_excel.xlsx')
    DSM = np.array(df)
    print(DSM)

    plt.matshow(DSM)
    plt.show()


    from PIL import Image
    srcImage = Image.open("Common/maps/Berlin_1_256.png")
    img1 = np.array(srcImage.convert('L').resize((150, 150)))
    img2 = 255-img1
    obs = np.where(img2==255)
    not_obs = np.where(img2<255)
    img3 = img2
    img3[obs]=1
    img3[not_obs]=0
    plt.matshow(img2)
    plt.show()

    from PIL import Image
    srcImage = Image.open("Common/maps/Boston_0_256.png")
    img1 = np.array(srcImage.convert('L').resize((150, 150)))
    img2 = 255-img1
    obs = np.where(img2==255)
    not_obs = np.where(img2<255)
    img3 = img2
    img3[obs]=1
    img3[not_obs]=0
    plt.matshow(img2)
    plt.show()

    from PIL import Image
    srcImage = Image.open("Common/maps/Paris_1_256.png")
    img1 = np.array(srcImage.convert('L').resize((150, 150)))
    img2 = 255-img1
    obs = np.where(img2==255)
    not_obs = np.where(img2<255)
    img3 = img2
    img3[obs]=1
    img3[not_obs]=0
    plt.matshow(img2)
    plt.show()


    from PIL import Image
    import cv2
    srcImage = Image.open("../maps/Berlin_1_256.png")
    plt.matshow(srcImage)
    plt.show()
    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    plt.matshow(img1)
    plt.show()
    img2 = cv2.bitwise_not(img1)
    obsticals=cv2.inRange(img2,250,255)
    plt.matshow(obsticals)
    plt.show()
    c, _ = cv2.findContours(obsticals,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255,255,255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges>0]=1

    plt.matshow(thicken_obs_and_edges)
    plt.show()

def get_DSM_berlin():
    srcImage = Image.open("Common/maps/Berlin_1_256.png")
    #srcImage = Image.open("../maps/Berlin_1_256.png")
    #srcImage = Image.open('../Common/maps/Berlin_1_256.png')
    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    img2 = cv2.bitwise_not(img1)
    obsticals = cv2.inRange(img2, 250, 255)
    c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    return thicken_obs_and_edges

def get_DSM_Paris():
    srcImage = Image.open("Common/maps/Paris_1_256.png")
    #srcImage = Image.open("../maps/Paris_1_256.png")
    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    img2 = cv2.bitwise_not(img1)
    obsticals = cv2.inRange(img2, 250, 255)
    c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    return thicken_obs_and_edges

def get_DSM_Boston():
    srcImage = Image.open("Common/maps/Boston_0_256.png")
    #srcImage = Image.open("../maps/Boston_0_256.png")
    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    img2 = cv2.bitwise_not(img1)
    obsticals = cv2.inRange(img2, 250, 255)
    c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    return thicken_obs_and_edges

if __name__ == '__main__':
    temp()