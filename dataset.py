import cv2
import glob

Dataset = 'BlurFace'
image_list = []
for filename in glob.glob(r'../' + Dataset + r'/img/*.jpg'): 
    img=cv2.imread(filename)
    image_list.append(img)
groundTruthDir = r"../"+Dataset+"/groundtruth_rect.txt" 

groundTruth = []
with open(groundTruthDir, 'r') as f:
    line = f.readline()
    while(line):
        line = line[:-1]
        if(Dataset == 'Basketball' or Dataset == 'DragonBaby' or Dataset == 'MountainBike'):
            line = list(map(int, line.split(',')))
        else:
            line = list(map(int, line.split('\t')))
        groundTruth.append(line)
        line = f.readline()
image = image_list[0]
W = groundTruth[0][2]
H = groundTruth[0][3]
start_point = (groundTruth[0][0], groundTruth[0][1])
end_point = (groundTruth[0][0] + W, groundTruth[0][1] + H)
color = (255, 0, 0)
thickness = 2
#image = cv2.rectangle(image, start_point, end_point, color, thickness)
frame = image[start_point[1] + thickness:end_point[1] - thickness, start_point[0] + thickness:end_point[0] - thickness]
leftW = W//2
rightW = (image.shape[1]) - W//2
upH = H//2
downH = (image.shape[0]) - H//2
K = 1