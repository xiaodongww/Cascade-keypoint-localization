import os
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

def draw_pic(img_path, points):
    im = plt.imread(img_path)
    dpi=80
    height, width, depth = im.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    implot = plt.imshow(im)
    for point in points:
        if point[-1] == -1:
            pass
        elif point[-1] == 0:
            plt.scatter([point[0]], [point[1]], c='r')
        elif point[-1] == 1:
            plt.scatter([point[0]], [point[1]], c='b')

    save_path = img_path.replace('Images', 'doted_Images')
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


ROOT = '/home/wuxiaodong/fa/'
# with open(os.path.join(ROOT, 'train/Annotations/train.csv')) as f:
with open(os.path.join(ROOT, 'models/test/result_testb.csv')) as f:
    line = f.readline()
    item_names = [item.strip() for item in line.split(',')]
    csv_content = []
    for line in f:
        items = []
        points = []
        for item in line.split(','):
            if '_' not in item:
                items.append(item)
            else:
                location = [int(i.strip()) for i in item.split('_')]
                items.append(location)
                points.append(location)
        onepic = {key:value for key,value  in zip(item_names, items)}
        draw_pic(os.path.join(ROOT, 'testb', onepic['image_id']), points)
        csv_content.append(onepic)






# im = array(Image.open('E:\\imagelocation\\6.jpg'))
# imshow(im)
# x = [10, box[1][0], box[2][0], box[3][0], ]
# y = [box[0][1], box[1][1], box[2][1], box[3][1], ]
# plot(x, y, 'r*')
# title('Plotting: "empire.jpg"')
# show()
