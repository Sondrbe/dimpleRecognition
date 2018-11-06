# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:05:33 2018

@author: sondrbe
"""

imgPath = r"C:\Users\Sondrbe\Documents\Dimple_Recognition\Dimples_Picture\png\dimple5.png"

from sklearn.cluster import KMeans
import cv2 #openCV
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

img = change_contrast(Image.open(imgPath), 100)
img = np.array(img)

lowerLimit = 100
upperLimit = 200

#img = cv2.imread(imgPath, 0)
edges = cv2.Canny(img, upperLimit, lowerLimit)

edges = edges / np.amax(edges)
        
edges_bool = edges.astype(bool)
dimples = np.invert(edges_bool).astype(int)

blobs, number_of_blobs = ndimage.label(dimples)


plt.figure()
plt.imshow(blobs)

plt.figure()
plt.imshow(dimples)



image = dimples
from scipy import ndimage

from skimage.morphology import watershed
from skimage.feature import peak_local_max


distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()





# Read image
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# Parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1500
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = True
params.minConvexity = 0.9
params.filterByInertia = True
params.minInertiaRatio = 0.7
params.minDistBetweenBlobs = 10
params.filterByColor = False
params.blobColor = 255

is_v2 = cv2.__version__.startswith("2.")
if is_v2:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)
    
    
img = change_contrast(Image.open(imgPath), 100)
img = np.array(img)
lowerLimit = 100
upperLimit = 200
#img = cv2.imread(imgPath, 0)
edges = cv2.Canny(img, upperLimit, lowerLimit)
edges = edges / np.amax(edges)        
edges_bool = edges.astype(bool)
dimples = np.invert(edges_bool).astype(int)    
# Detect blobs.
dimples = np.array(dimples*255, np.uint8)
keypoints = detector.detect(dimples)
print(keypoints)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(dimples, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

plt.imshow(im_with_keypoints)

plt.imshow(dimples, cmap='gray')






plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])





















#--------------------------------------------

import numpy as np
from skimage import data
coins = data.coins()
histo = np.histogram(coins, bins=np.arange(0, 256))
plt.plot(histo[0])

from skimage.feature import canny
edges = canny(coins/255.)
plt.imshow(edges, 'gray')

from scipy import ndimage
fill_coins = ndimage.binary_fill_holes(edges)
label_objects, nb_labels = ndimage.label(fill_coins)
plt.imshow(fill_coins, 'gray')
plt.imshow(label_objects, 'gray')

sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

from skimage.filters import sobel
elevation_map = sobel(coins)
plt.imshow(elevation_map, 'gray')

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)

segmentation = ndimage.binary_fill_holes(segmentation - 1)

labeled_coins, _ = ndimage.label(segmentation)


#--------------------------------------------


"""
This is actually very useful code
"""

img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
histo = np.histogram(img, bins=np.arange(0, 256))
plt.plot(histo[0])
edges = canny(img/255.)
plt.imshow(edges)
fill_dimples = ndimage.binary_fill_holes(edges)
label_objects, nb_labels = ndimage.label(fill_dimples)
plt.imshow(fill_dimples)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2
elevation_map = sobel(dimples)
plt.imshow(elevation_map)
markers = np.zeros_like(dimples)
markers[dimples < 0.5] = 1
markers[dimples > 0.5] = 2
segmentation = watershed(elevation_map, markers)
plt.imshow(segmentation)
#segmentation = ndimage.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndimage.label(segmentation)
plt.imshow(labeled_coins)




kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
dilated = cv2.dilate(image, kernel)
_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

test = ndimage.filters.gaussian_filter(dimples, sigma=5)
plt.imshow(test)
edges = canny(img/255.)
plt.imshow(edges)















#--------------------------------------------------------------------------------
#  The correct code for dimple tracing
#--------------------------------------------------------------------------------
import cv2
import numpy as np
from sklearn.cluster import KMeans
import cv2 #openCV
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

imgPath = r"C:\Users\Sondrbe\Documents\Dimple_Recognition\Dimples_Picture\png\dimple5.png"

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

img = change_contrast(Image.open(imgPath), 100)
img = np.array(img)

lowerLimit = 100
upperLimit = 200

#img = cv2.imread(imgPath, 0)
edges = cv2.Canny(img, upperLimit, lowerLimit)
edges = edges / np.amax(edges)   
edges = edges.astype(np.uint8)     
#edges_bool = edges.astype(bool)
#dimples = np.invert(edges_bool).astype(np.uint8)

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)
ax1.imshow(edges, 'gray')

# Erosion: A px will be considered 1 only if all px's in the kernel is 1.
kernel = np.ones((2,2),np.uint8)
erosion2 = cv2.erode(edges,kernel,iterations = 1)
ax2.imshow(erosion2, 'gray')

kernel = np.ones((4,4),np.uint8)
erosion3 = cv2.erode(edges,kernel,iterations = 1)
ax3.imshow(erosion3, 'gray')

kernel = np.ones((6,6),np.uint8)
erosion4 = cv2.erode(edges,kernel,iterations = 1)
ax4.imshow(erosion4, 'gray')

"""
I would almost say a kernel of 2*2 or 3*3 is perfect!
"""

# Dilation: A px will be considered 1 if one or more px's in the kernel is 1.
#dilation = cv2.dilate(img,kernel,iterations = 1)



fig = plt.figure()
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)
ax1.imshow(edges, 'gray')

# Dilation: A px will be considered 1 if one or more px's in the kernel is 1.
kernel = np.ones((2,2),np.uint8)
dilation2 = cv2.dilate(edges,kernel,iterations = 1)
ax2.imshow(dilation2, 'gray')

kernel = np.ones((3,3),np.uint8)
dilation3 = cv2.dilate(edges,kernel,iterations = 1)
ax3.imshow(dilation3, 'gray')

kernel = np.ones((6,6),np.uint8)
dilation4 = cv2.dilate(edges,kernel,iterations = 1)
ax4.imshow(dilation4, 'gray')


ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)
ax1.imshow(dilation4, 'gray')

# Erosion: A px will be considered 1 only if all px's in the kernel is 1.
kernel = np.ones((2,2),np.uint8)
erosion2 = cv2.erode(dilation4, kernel,iterations = 1)
ax2.imshow(erosion2, 'gray')

kernel = np.ones((4,4),np.uint8)
erosion3 = cv2.erode(dilation4, kernel,iterations = 1)
ax3.imshow(erosion3, 'gray')

kernel = np.ones((6,6),np.uint8)
erosion4 = cv2.erode(dilation4, kernel,iterations = 1)
ax4.imshow(erosion4, 'gray')



"""
I can not fill the edges! I must fill the dimples.
If i fill the edges; the entire image should become 1's, as they are all connected.
"""
edges_bool = erosion2.astype(bool)
dimples = np.invert(edges_bool).astype(np.uint8)
fill_dimples = ndimage.binary_fill_holes(dimples)
label_objects, num_dimples = ndimage.label(fill_dimples)

plt.imshow(fill_dimples, 'gray')
plt.imshow(label_objects, 'gray')

#plt.imshow(erosion2, 'plasma')


"""
The label_objects variable is perfect!!!
I should create functions to get statistics for each dimple from this one.


test = np.zeros(np.shape(img))
test[indices] = 1
plt.imshow(test, 'gray')

"""

from numba import jit

#@jit(nopython=True)
def getProperties(label, label_objects):
    indices = np.where(label_objects == label)
    area = len(indices[0])
    width = max(indices[0])-min(indices[0])
    height = max(indices[1])-min(indices[1])
    return area, width, height

getProperties(700, label_objects)



#@jit(nopython=True)
def getInfo(label_objects):
    areaList, widthList, heightList = [],[],[]
    for dimple in range(1,num_dimples+1):
        area, width, height = getProperties(dimple, label_objects)
        areaList.append(area)
        widthList.append(width)
        heightList.append(height)
    return areaList, widthList, heightList
        
test = getInfo(label_objects)        
        

hist, bin_edges = np.histogram(test[0], bins=range(num_dimples))
#plt.hist(hist,bins='auto')
#y = np.sort(hist)
plt.plot(hist)


"""
SUMMARY:
    
Start with edge-detection.
Then, dilate the edges. They must be thicker, straight away.
Find the optimal dilation kernel. One 3*3 is the same as two 2*2. 
    So only perform this dilation once, but find the optimal kernel.
Afterwards, might perform a erosion to make the lines thinner.
    As they now have connected, they will most likely stay connected.
    BUT, be very careful, and only use a 2*2 erosion if you so chose.
Then, invert the edges to obtain dimples, and fill them.
    Use ndi.label() to get their px's and the count of dimples.
Easy to then get Area, Width, Heigth etc.
Find a clever way to incorporate




Other interesting properties:
    
Convexity measure:
    Convexity_measure(Shape) = Area(Shape) / Area(ConvexHull(Shape))
    
    
"""







#--------------------------------------------------------------------------------
#  The correct code for dimple tracing
#--------------------------------------------------------------------------------
import cv2
import numpy as np
from sklearn.cluster import KMeans
import cv2 #openCV
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

imgPath = r"C:\Users\Sondrbe\Documents\Particle_Images\6061-cast-01.tif"

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

img = change_contrast(Image.open(imgPath), 100)
img = np.array(img)

resolution = np.shape(img)
img2 = img[:int(92/100*resolution[0]), :]
plt.imshow(img)


#-------------------- Edge tracing -------------------

lowerLimit = 100
upperLimit = 200

edges = cv2.Canny(img, upperLimit, lowerLimit)
edges = edges / np.amax(edges)   
edges = edges.astype(np.uint8)    
plt.imshow(edges, 'gray')

#plt.figure()
#kernel = np.ones((3,3),np.uint8)
#closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#plt.imshow(closing)
#fill_particles = ndi.binary_fill_holes(closing)
#plt.imshow(fill_particles)
#labeled_particles, num_particles = ndi.label(closing)
#plt.imshow(labeled_particles) 

#edges_bool = edges.astype(bool)
#dimples = np.invert(edges_bool).astype(np.uint8)

#--- Dilate the images, to close the edges
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
plt.imshow(dilation, 'gray')

#--- Fill the particles --------
edges_bool = dilation.astype(bool)
fill_particles = ndimage.binary_fill_holes(edges_bool)
fill_particles = fill_particles.astype(np.uint8)
plt.imshow(fill_particles, 'gray')

#--- Erode the particles with the corresponding kernel size used for dilation,
#--- to retrieve the original size of the particles
#kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(fill_particles, kernel, iterations = 1)
plt.imshow(erosion, 'gray')


#--- Label each unique particle
labeled_particles, num_particles = ndimage.label(erosion)
plt.imshow(labeled_particles)



"""
I will now write functions to obtain the center of a particle
"""
def centersParticle():
    centers = []
    for num in range(1, num_particles+1):
        indices = np.where(labeled_particles==num)
        x_pos = np.average(indices[1])
        y_pos = np.average(indices[0])
        centers.append((x_pos,y_pos))
    return centers

centers = centersParticle()
    


        