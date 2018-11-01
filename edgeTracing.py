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
    











"""
For each cluster, I should calibrate a probability distribution.
Outputs are direction, distance and area, so these must be random variables.


Write some code that assigns an element the corresponding particle size.
Use spatial coordinates to assign the average value inside the kernel.
Simply use average pixel values, i.e Area(particle)/Area(kernel).

Write some code that assigns the nearest dimple's void volume fraction as the 
failure/coalescence value. Use k-NN to get nearest dimple.

BUT, I guess I have to calibrate a representative model,
because I can not use axisymmetric specimens!
Need solid elements, and 3D distribution of particles and voids/dimples.





THUS, I should calibrate some distributions
    Distance: Normal distribution
    Angle: Normal distribution, perhaps sum of 3 individual distributions. 
        Divide angle%(2*np.pi), to get an angle inside the interval.
    Area: Normal distribution.
May actually use mean and standard deviation measures, no need for log-likelihood...

Could perhaps create 4 clusters inside a cluster, calculate 4 individual means 
and standard deviations, and randomly choose one of these functions to generate that 
cluster, each particle is another random choice.

Then all I would need to do is to generate clusters. They do seem quite uniformly
distributed. Perhaps try this?

I think that by breaking the problem up into clusters, I should obtain some 
of the same patterns as in the image....
"""









# STOCHASTIC MICROSTRUCTURE CHARACTERIZATION
size = (200,200)
img = np.zeros(size)

# Distribute some squares:
square = (3,1)
num_squares = 50
for n in range(num_squares):
    x_centre = np.random.randint(0,size[0])
    y_centre = np.random.randint(0,size[0])
    small_corner = (x_centre-square[0], y_centre-square[1])
    big_corner = (x_centre+square[0], y_centre+square[1])
    for ind1 in range(small_corner[0],big_corner[0]+1):
        for ind2 in range(small_corner[1], big_corner[1]+1):
            if ind1 in range(size[0]) and ind2 in range(size[1]):
                img[ind1, ind2] = 1
    
plt.imshow(img, 'gray')    


n1, n2 = size

# Choose neighborhood size:
w = 5
h = 5

# Generate the training set:
T = []
#T = np.zeros((n1-2*h)*(n2-2*w))
#for row in range((n1-2*h)*(n2-2*w)):
for pixel1 in range(h, n1-h):
    for pixel2 in range(w, n2-w):
        x_ij = img[pixel1, pixel2]
        subMatrix = img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
        N_ij_extra = subMatrix.flatten()
        M_ij = N_ij_extra[:(2*w+1)*h + w]
        N_ij = np.delete(N_ij_extra,[(2*w+1)*h + w + 1])
        D = (x_ij, M_ij[:])
        T.append(D)
        

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit([x[1] for x in T], [x[0] for x in T])

# To get predicted class (pixel value)
# clf.predict([[0]*12])
# To get predicted probability:
# clf.predict_proba([[0]*9+[1]*3])



# Assess performance with Cross-validation measures:


# Generate a new image, with correct void volume fraction:
def generateImage(size):
    # Initialize boundaries of a generated image:
    new_img = np.zeros((size[0]+2*w, size[1]+2*h))
    row1_ind = np.random.randint(size[0]-w)
    col1_ind = np.random.randint(size[1]-h)
    square1_ind = np.random.randint(size[1]-max(h,w))
    
    row_bott_ind = np.random.randint(size[0]-w)
    col_right_ind = np.random.randint(size[1]-h)
    square_upRight_ind = np.random.randint(size[1]-max(h,w))
    
    img_col1_slice = img[np.ix_(range(row1_ind, row1_ind+w), range(0,size[1]))]
    img_row1_slice = img[np.ix_(range(0,size[1]), range(col1_ind, col1_ind+h))]
    img_square1_slice = img[np.ix_(range(square1_ind,square1_ind+w), range(square1_ind,square1_ind+h))]
    
    new_img[range(w), h:-h] = img_col1_slice
    new_img[w:-w, range(h)] = img_row1_slice
    new_img[:w,:h] = img_square1_slice
    
    plt.imshow(new_img)
    
    c = 0.
    TOL = 0.04
    while True:
        work_img = copy.deepcopy(new_img)
        for pixel1 in range(w, size[0]+w):
            for pixel2 in range(h, size[1]+h):
                subMatrix = work_img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
                N_ij_extra = subMatrix.flatten()
                M_ij = N_ij_extra[:(2*w+1)*h + w]
                p_ij = clf.predict_proba([M_ij])[0][1]
                p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                work_img[pixel1, pixel2] = x_ij
        VF = np.sum(work_img)/np.sum(img)
        if (VF < 1): 
            VF = 1 / VF
            c += 0.005
        else: c -= 0.005
        print(VF)
        if (VF-1 <= TOL): break
        print('An image is generated...')
    final_img = work_img[h:-h, w:-w]
    return final_img

fig, [ax1,ax2] = plt.subplots(1,2)
ax1.set_title('Image from experiments')
ax1.imshow(img, 'gray')
ax2.set_title('Generated image')
ax2.imshow(new_img, 'gray')
ax2.imshow(final_img, 'gray')




# Define some performance functions, to choose the optimal neighborhood size:
# This is simply something to measure how close different images are.
# Start with a large neighborhood, and keep decreasing it until the images 
# likeliness starts to diverge, or opposite.
def TPCF(image):
    n1, n2 = np.shape(image)
    R = np.zeros((n1,n2))
    for dx in range(n1):
        for dy in range(n2):
            for pixel1 in range(n1):
                for pixel2 in range(n2):
                    image[pixel1, pixel2] * image[pixel1+dx, pixel2+dy]
            R[pixel1, pixel2] = 
            
            
            
            
import numpy as np            
x,y = np.mgrid[0:n1,0:n2]
x = x.flatten()
y = y.flatten()
z1 = img.flatten()
z2 = final_img.flatten()
from halotools.mock_observables import tpcf
            


sample1 = np.vstack((x,y,z1)).T
sample2 = np.vstack((x,y,z2)).T
rbins = np.linspace(1,40,51)
#rbins = np.logspace(-1, 1, 10)

twoPointCrossCorr = tpcf(sample1, rbins, sample2, do_auto=False, period=n1*n2)
#twoPointCrossCorr = tpcf(sample1, np.logspace(-1, 1, 10), period=n1*n2)
plt.plot(twoPointCrossCorr)


"""
I think I know how to generate 3D images.
Create all 3 boundaries, in 3D space. For each pixel, calculate the probability 
of a 1 pixel, and average these 3 probabilities.
Then generate the box with distributions!

When plotting the results, I could plot each as a cube in a 3D plot.
Pixel 1 will range between (0,0,0), (1,0,0), (0,1,0), (1,1,0), and for z=1 as well.
Could draw a surface plot here, as a hyperplane...
"""

# Perhaps generate a lot of pictures, and stack them according to a minimum 
# MSE error?
def Loss(images):
    totMSE = 0
    for image1,image2 in zip(images[::-1], images[1::]):
        mse = ((img1 - img2) ** 2).mean(axis=None)
        totMSE += mse
    return totMSE

# A random slice through the generated 3D object should have similar statistical 
# attributes/values as the input images.
# I guess I could keep reordering them until the same statistical values are obtained.
# Thus I need a measure for statistical similarity.
    
"""
I should simply generate a lot of images, and stack them in a different order.
I should have a statistical measure, to ensure that the slices of the other 
dimensions preserves the statistical distribution.
I could e.g. use Genetic Algorithm to try to minimize the differences in statistical
distribution, by re-ordering the image stack!

Or, perhaps generate three 3D blocks, and take some mean value in each pixel..

Or, order one stack to prefer clusters?
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.mgrid[0:5,0:5]
x = x.flatten(); y = y.flatten()
z = np.random.randint(low=0, high=10, size=(len(x.flatten()),))

ax.plot(x,y,z, 'bo')










"""
How to generate boundaries for a new image:
"""
import numpy as np
img1, img2, img3 = img,img,img

# Should definetely use a square neighborhood, as the material is isotropic and all..
if w==h: n = h
else: n=h

# Should also generate a square box, of size equal to box_shape:
n1 = 200
n2 = 200
n3 = 200
size = (n1, n2, n3)

# Initialize boundaries of a generated image:
new_img = np.zeros((size[0]+2*n, size[1]+2*n, size[2]+2*n))
boundaryImgSize1 = (size[0]+n+1, size[1]+n+1)
boundaryImgSize2 = (size[1]+n+1, size[2]+n+1)
boundaryImgSize3 = (size[0]+n+1, size[2]+n+1)
boundaryImg1 = generateImage(boundaryImgSize1)
boundaryImg2 = generateImage(boundaryImgSize2)
boundaryImg3 = generateImage(boundaryImgSize3)
# Must initialize n rows of px's in each dimension. All outer rows are only 0's, 
# the inner is 6 generated images in correct size!
new_img[n-1,n-1:,n-1:] = boundaryImg2
new_img[n-1:,n-1:,n-1] = boundaryImg1
new_img[n-1:,n-1,n-1:] = boundaryImg3

"""
final_img = new_img[2*n:, 2*n:, 2*n:]
"""






"""
How to plot the generated particles in 3D space:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_cube(cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    #ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    #ax.set_aspect('equal')
    return



def plotParticles3D(image):
    n1,n2,n3 = np.shape(image)
    for px1 in range(0,n1):
        for px2 in range(0,n2):
            for px3 in range(0,n3):
                if image[px1,px2,px3] == 1:
                    cube_definition = [
                            (px1,px2,px3), (px1+1,px2,px3), 
                            (px1,px2+1,px3), (px1,px2,px3+1)]     
                    plot_cube(cube_definition)
    ax.set_xlim(0, n1)
    ax.set_ylim(0, n2)
    ax.set_zlim(0, n3)                    
    return
                    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotParticles3D(new_img)

