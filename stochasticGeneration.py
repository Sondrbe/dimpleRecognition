"""
Stochastic microstructure characterization and reconstruction via supervised learning
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import copy

"""
 Create an example image, to use in the calibrations during the alpha version:
"""
size = (200,200)
img = np.zeros(size)

# Distribute some squares:
square = (2,2)
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


"""
Create the training set, and calibrate the supervised model
"""
# Choose neighborhood size:
w = 5
h = 5

n1, n2 = size
# Generate the training set:
T = []
for pixel1 in range(h, n1-h):
    for pixel2 in range(w, n2-w):
        x_ij = img[pixel1, pixel2]
        subMatrix = img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
        N_ij_extra = subMatrix.flatten()
        M_ij = N_ij_extra[:(2*w+1)*h + w]
        # Delete pixel x_ij from the neighborhood, as it doesn't count:
        N_ij = np.delete(N_ij_extra,[(2*w+1)*h + w + 1])
        D = (M_ij[:], x_ij)
        T.append(D)
    
# Calibrate the model:
clf = tree.DecisionTreeClassifier()
clf = clf.fit([x[0] for x in T], [x[1] for x in T])
# To get predicted class (pixel value)
# clf.predict([[0]*12])
# To get predicted probability:
# clf.predict_proba([[0]*9+[1]*3])

"""

# Generate a new image, with correct void volume fraction:
def generateImage(size, clf):
    import copy
    # Initialize boundaries of a generated image:
    #new_img = copy.deepcopy(img)
    #new_img[w:-w,h:-h] = 0
    
    # I could eventually do something like this:
    #new_img = np.zeros((size[0]+4*w, size[1]+4*h))
    
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
    
    
    #plt.imshow(new_img)
    
    c = 0.
    TOL = 0.1
    while True:
        work_img = copy.deepcopy(new_img)
        for pixel1 in range(w, size[0]):
            for pixel2 in range(h, size[1]):
                subMatrix = work_img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
                N_ij_extra = subMatrix.flatten()
                M_ij = N_ij_extra[:(2*w+1)*h + w]
                p_ij = clf.predict_proba([M_ij])[0][1]
                p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                work_img[pixel1, pixel2] = x_ij
        
        
        VF = np.sum(work_img)/np.sum(img)
        print(VF)
        if (VF < 1): 
            c += 0.05
            if (abs(VF-1) <= TOL): break
        else: 
            c -= 0.01
            if (VF-1 <= TOL): break
        #plt.imshow(work_img)    
    final_img = work_img[h:-h, w:-w]
    plt.imshow(final_img)
    return final_img

size = (200,200)
final_img = generateImage(size, clf)
plt.imshow(final_img)

fig, [ax1,ax2] = plt.subplots(1,2)
ax1.set_title('Image from experiments')
ax1.imshow(img, 'gray')
ax2.set_title('Generated image')
ax2.imshow(new_img, 'gray')
ax2.imshow(final_img, 'gray')

"""








"""
Perhaps the best code so far, for generating images!
"""
# Generate image from zeroes, less efficient:
def generateImageFrom0s(size, clf, orig_img=[]):
    """
    An image with 4*w at each side, as well as 4*h at both top/bottom are 
    generated, looped through, and the image of shape (size) is 
    returned from the center.
    Probabilities are adjusted until the same void volume fraction is obtained 
    if an comparison image is given as an optional argument.
    """
    # Initialize boundaries of a generated image:
    new_img = np.zeros((size[0]+8*w, size[1]+8*h))
        
    c = 0.
    TOL = 0.15
    while True:
        work_img = copy.deepcopy(new_img)
        for pixel1 in range(h, size[0]+4*h):
            for pixel2 in range(w, size[1]+7*w):
                subMatrix = work_img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
                N_ij_extra = subMatrix.flatten()
                M_ij = N_ij_extra[:(2*w+1)*h + w]
                p_ij = clf.predict_proba([M_ij])[0][1]
                p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                work_img[pixel1, pixel2] = x_ij      
        if len(orig_img) == 0: 
            print('Generated an image')
            break
        else: # Perform the void volume fraction calibration:
            VF = np.sum(work_img[4*h:-4*h, 4*w:-4*w])/np.sum(orig_img)
            print(VF)
            if (VF < 1): 
                c += 0.005
                if (abs(VF-1) <= TOL): break
            else: 
                c -= 0.005
                if (VF-1 <= TOL): break   
    final_img = work_img[4*h:-4*h, 4*w:-4*w]
    #plt.imshow(final_img)
    return final_img

size = (200,200)
final_img = generateImageFrom0s(size, clf, img)

# Plot the generated image(s):
num_images = 4

ax = plt.subplot(1,num_images+1, 1)
ax.set_title('Image from experiments')
ax.imshow(img, 'gray')
for i in range(num_images):
    ax = plt.subplot(1,num_images+1, i+2)
    ax.set_title('Generated image')
    ax.imshow(generateImageFrom0s(size, clf, img), 'gray')

"""


# Assess performance with Cross-validation measures:
"""
from sklearn.model_selection import train_test_split
# Sample a training set while holding out e.g 20% of the data for testing:
x_data = [x for x,_ in T]
y_data = [y for _,y in T]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
# Calibrate the model:
clf.fit(X_train, y_train)
# Assess performance:
clf.score(X_test, y_test) 
clf.score(X_train, y_train) 
"""
I should perhaps plot the accuracy from training and test set, as I keep 
pruning the tree. Ideally, the accuracy on the training set will decrease while 
the test accuracy doesn't. When the test accuracy starts to decrease, 
I must stop pruning the model. Everything up until that point is simply great 
for generalization to unseen problems...
"""

# Cross-validation:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, x_data, y_data, cv=5)
print(scores)
print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

# Different arguments to the DecisionTreeClassifier:
criterion = 'gini' # 'entropy'
splitter = 'best' # 'random'
max_depth = None # integer, e.g. 20
min_samples_split = 2 # integer, e.g. 4
min_samples_leaf = 1 # integer, e.g. 2
min_weight_fraction_leaf = 0. # float
max_features = None # 'auto', 'sqrt', 'log2'
random_state = None #integer
max_leaf_nodes = None # integer
min_impurity_decrease = 0. #float
class_weight = None # dict
presort = False # Bool

clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
       min_weight_fraction_leaf=min_weight_fraction_leaf, 
       max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, 
       min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, 
       presort=presort)
"""
Default parameters leads to unpruned trees. Try tweaking 'max_depth' and 
'min_samples_leaf' etc to prune them. 


Decision tree attributes:
classes_ : array of shape = [n_classes] or a list of such arrays
    The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).
feature_importances_ : array of shape = [n_features]
    Return the feature importances.
max_features_ : int,
    The inferred value of max_features.
n_classes_ : int or list
    The number of classes (for single output problems), or a list containing the number of classes for each output (for multi-output problems).
n_features_ : int
    The number of features when fit is performed.
n_outputs_ : int
    The number of outputs when fit is performed.
tree_ : Tree object
    The underlying Tree object. Please refer to help(sklearn.tree._tree.Tree) for attributes of Tree object and Understanding the decision tree structure for basic usage of these attributes.    
"""




"""
# Define a performance measure function
"""

import numba
@numba.jit(nopython=True, locals={'distInd': numba.int32, 'max_dist':numba.int32})
def twoPointLinealPathCorrelation(image, max_dist):
    """
    Improvement here:
        Instead of sampling a lot of Monte Carlo points, simply use brute-force to 
        calculate the correlation in the entire image. This will be much faster
        and more accurate.
    """
    samePhase = np.zeros(max_dist)
    totalExecuted = np.zeros(max_dist)    
    for row1 in range(max_dist, image.shape[0]-max_dist):
        for col1 in range(max_dist, image.shape[0]):
            for row2 in range(row1-max_dist, row1+max_dist):
                for col2 in range(col1-max_dist, col1+max_dist):
                    dist = ((row1-row2)**2 + (col1-col2)**2)**0.5
                    if dist <= max_dist:
                        distInd = int(dist)
                        totalExecuted[distInd] += 1
                        if image[row1,col1] == 1:
                            if image[row2, col2] == 1:
                                samePhase[distInd] += 1  
    return samePhase/totalExecuted


max_dist = 25 #px's
performance = twoPointLinealPathCorrelation(final_img, max_dist)
plt.plot(range(len(performance)), performance)
plt.plot(range(len(performance)), performance, 'k*')

# Start with a large neighborhood, and keep decreasing it until the images 
# likeliness starts to diverge, or opposite...















"""
I think I know how to generate 3D images.
Create all 3 boundaries, in 3D space. For each pixel, calculate the probability 
of a 1 pixel, and average these 3 probabilities.
Then generate the box with distributions!
"""
# Should definetely use a square neighborhood, as the material is isotropic and all..
if w==h: n = h
else: n=h

# Should also generate a square box, of size equal to box_shape:
n1 = 50
n2 = 50
n3 = 50
size = (n1, n2, n3)

# Initialize boundaries of a generated image:
new_img = np.zeros((size[0]+2*n, size[1]+2*n, size[2]+2*n))
boundaryImgSize1 = (size[0]+n+1, size[1]+n+1)
boundaryImgSize2 = (size[1]+n+1, size[2]+n+1)
boundaryImgSize3 = (size[0]+n+1, size[2]+n+1)
boundaryImg1 = generateImageFrom0s(boundaryImgSize1, clf)
boundaryImg2 = generateImageFrom0s(boundaryImgSize2, clf)
boundaryImg3 = generateImageFrom0s(boundaryImgSize3, clf)
# Must initialize n rows of px's in each dimension. All outer rows are only 0's, 
# the inner is 6 generated images in correct size!
new_img[n-1,n-1:,n-1:] = boundaryImg2
new_img[n-1:,n-1:,n-1] = boundaryImg1
new_img[n-1:,n-1,n-1:] = boundaryImg3




"""
Now, the 3D distribution of particles are generated:
        new_img = np.zeros((size[0]+8*w, size[1]+8*h))
        
    c = 0.
    TOL = 0.15
    while True:
        work_img = copy.deepcopy(new_img)
        for pixel1 in range(h, size[0]+4*h):
            for pixel2 in range(w, size[1]+7*w):
                subMatrix = work_img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
                N_ij_extra = subMatrix.flatten()
                M_ij = N_ij_extra[:(2*w+1)*h + w]
                p_ij = clf.predict_proba([M_ij])[0][1]
                p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                work_img[pixel1, pixel2] = x_ij      
        if len(orig_img) == 0: 
            print('Generated an image')
            break
        else: # Perform the void volume fraction calibration:
            VF = np.sum(work_img[4*h:-4*h, 4*w:-4*w])/np.sum(orig_img)
            print(VF)
            if (VF < 1): 
                c += 0.005
                if (abs(VF-1) <= TOL): break
            else: 
                c -= 0.005
                if (VF-1 <= TOL): break   
    final_img = work_img[4*h:-4*h, 4*w:-4*w]
"""

def add(img):
    return clf.max_depth

@numba.jit(nopython=True)
def myFunc():
    num = 0
    for i in range(100):
        num += add(img)
    return num
myFunc()
        




@numba.jit(nopython=True)
def generate3dDistribution(new_img, clf):
    c = 0.
    VVF_expected = 0.001
    TOL = 0.2
    tot_pxs = n1*n2*n3
    curr_px = 0
    iteration = -1
    while True:
        iteration += 1
        work_img = copy.deepcopy(new_img)
        for px1 in range(n, size[0]+n):
            for px2 in range(n, size[1]+n):
                for px3 in range(n, size[2]+n):
                    curr_px += 1
                    slice1 = work_img[px1,:,:]
                    slice2 = work_img[:,:,px3]
                    slice3 = work_img[:,px2,:]
                    subMatrix1 = slice1[np.ix_(range(px2-n,px2+n+1), range(px3-n,px3+n+1))]
                    subMatrix2 = slice2[np.ix_(range(px1-n,px1+n+1), range(px2-n,px2+n+1))]
                    subMatrix3 = slice3[np.ix_(range(px1-n,px1+n+1), range(px3-n,px3+n+1))]
                    N_ij_extra_1 = subMatrix1.flatten()
                    N_ij_extra_2 = subMatrix2.flatten()
                    N_ij_extra_3 = subMatrix3.flatten()
                    M_ij_1 = N_ij_extra_1[:(2*n+1)*n + n]
                    M_ij_2 = N_ij_extra_2[:(2*n+1)*n + n]
                    M_ij_3 = N_ij_extra_3[:(2*n+1)*n + n]
                    p_ij_1 = clf.predict_proba([M_ij_1])[0][1]
                    p_ij_2 = clf.predict_proba([M_ij_2])[0][1]
                    p_ij_3 = clf.predict_proba([M_ij_3])[0][1]
                    p_ij = (p_ij_1 + p_ij_2 + p_ij_3)/3.
                    
                    p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                    x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                    work_img[px1, px2, px3] = x_ij
                    
                    # Tell the user how much calculations that has been performed:
                    if curr_px % int(tot_pxs/30) == 0:
                        print('Iteration '+str(iteration)+':', str(curr_px / tot_pxs * 100) + '%')
        VF = np.sum(work_img[n:-n, n:-n, n:-n])/tot_pxs
        if (VF < 1): 
            c += 0.005
            if (abs(VF-1) <= TOL): break
        else: 
            c -= 0.005
            if (VF-1 <= TOL): break   
    final_img = work_img[n:-n, n:-n, n:-n]                    
    return final_img

final_3D_Image = generate3dDistribution(new_img, clf)


"""
How to plot the generated particles in 3D space:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_cube(cube_definition):
    """
    This function plots a single cube in 3D.
    Input: The 4 coordinates that are needed in order to define a 3D cube.
    Output: Plots this cube to the existing ax object...
    """
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
    return



def plotParticles3D(image3D):
    """
    This function loops through a 3D box, and plots all particles.
    Input: A generated 3D image
    Output Plots all particles to the existing ax object...
    """
    n1,n2,n3 = np.shape(image3D)
    px_count = 0
    for px1 in range(0,n1):
        for px2 in range(0,n2):
            for px3 in range(0,n3):
                px_count += 1
                if image3D[px1,px2,px3] == 1:
                    cube_definition = [
                            (px1,px2,px3), (px1+1,px2,px3), 
                            (px1,px2+1,px3), (px1,px2,px3+1)]     
                    plot_cube(cube_definition)
        print('%.2f'%(px_count/(n1*n2*n3)*100), '%')
    ax.set_xlim(0, n1)
    ax.set_ylim(0, n2)
    ax.set_zlim(0, n3)                    
    return
                    

"""
In order to plot the generated 3D image:
"""
for px1 in np.random.randint(0,n1,3):
    for px2 in np.random.randint(0,n2,3):
        for px3 in np.random.randint(0,n3,3):
            new_img[px1,px2,px3] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plotParticles3D(final_3D_Image)








# Check what scipy classification tree does with data not encountered before:
"""
T = [([0,0,0,1], 0), 
     ([0,0,0,1], 0),
     ...
     ]
clf = tree.DecisionTreeClassifier()
clf = clf.fit([x[1] for x in T], [x[0] for x in T])
"""

T = [([0,0,0],0), ([0,0,0],0), ([0,0,0],0), ([0,0,0],0), ([0,0,0],1), ([0,0,0],1),
     ([0,0,1],0), ([0,0,1],0), ([0,0,1],0), ([0,0,1],0), ([0,0,1],0), ([0,0,1],1),
     ([0,1,0],0), ([0,1,0],0), ([0,1,0],0), ([0,1,0],0), ([0,1,0],0), ([0,1,0],0),
     ([0,1,1],0), ([0,1,1],0), ([0,1,1],0), ([0,1,1],0), ([0,1,1],1), ([0,1,1],1),
     ([1,0,0],0), ([1,0,0],0), ([1,0,0],0), ([1,0,0],0), ([1,0,0],0), ([1,0,0],0),
     ([1,0,1],0), ([1,0,1],0), ([1,0,1],0), ([1,0,1],0), ([1,0,1],1), ([1,0,1],1),
     ([1,1,0],0), ([1,1,0],0), ([1,1,0],1), ([1,1,0],1), ([1,1,0],1), ([1,1,0],1)]

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit([x[0] for x in T], [x[1] for x in T])

clf.predict_proba([[1,1,1]])
# This one predicts a probability even to unseen data!!!
# Even worse, this probability is unnaturally large for the wrong class!?

# My own decision tree, where unseen data is classified as background (or whatever):
probs = dict()
num_tot = dict()
for x_data, y_data in T:
    x_data = list(x_data.astype(int))
    if str(x_data) not in probs.keys():
        probs[str(x_data)] = 0
        num_tot[str(x_data)+'_num'] = 0
        num_tot[str(x_data)+'_tot'] = 0
    if y_data == 0: num_tot[str(x_data)+'_num'] += 1
    num_tot[str(x_data)+'_tot'] += 1
for x,y in probs.items():
    probs[x] = num_tot[str(x)+'_num'] / num_tot[str(x)+'_tot']

# Different number of permutations with boolean px's are 2**n:
def classifierMissingData(classifierDict):
    key = next(iter(classifierDict))
    length = len(key)
    permutations = 2**length
    if len(classifierDict) < permutations:
        return False
    else: return True
    
    
def addMissingData(classifierDict, probs_value=0.):
    """
    Binary numbers are written with a leading 0b. Everything after that is the number.
    bin(number) returns a string of binary represntation.
    Simply writing 0b.... returns the integer straight away, no intermediary string..
    """
    neighborhood_size = len(T[0][0])
    for num in range(2**neighborhood_size):
        perm = bin(num)[2:]
        if len(perm) < neighborhood_size:
            missing = neighborhood_size - len(perm)
            perm = '0'*missing + perm
        perm = '['+ ', '.join(perm) + ']'
        if perm not in classifierDict.keys():
            classifierDict[perm] = probs_value
    return classifierDict

probs = addMissingData(probs, 0.1)    

        

# Generate a new image, with correct void volume fraction:
def generateImage(size):
    import copy
    # Initialize boundaries of a generated image:
    new_img = np.zeros((size[0]+2*w, size[1]+2*h))
    #new_img[2*w,10] = 1
    #new_img[2*w,100] = 1
    #new_img[2*w,150] = 1
    #new_img[10,2*h] = 1
    #new_img[100,2*h] = 1
    #new_img[150,2*h] = 1
    c = 0.
    TOL = 0.4
    while True:
        work_img = copy.deepcopy(new_img)
        for pixel1 in range(w, size[0]+w):
            for pixel2 in range(h, size[1]+h):
                subMatrix = work_img[np.ix_(range(pixel1-w,pixel1+w+1), range(pixel2-h,pixel2+h+1))]
                N_ij_extra = subMatrix.flatten()
                M_ij = N_ij_extra[:(2*w+1)*h + w]
                M_ij_copy = copy.deepcopy(M_ij)
                #if str(list(M_ij_copy.astype(int))) in probs.keys():
                #    p_ij = clf.predict_proba([M_ij])[0][1]
                #else: 
                #    p_ij = 0.01
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
        plt.imshow(work_img)
        if (VF-1 <= TOL): break
        print('An image is generated...')
    final_img = work_img[2*w:, 2*h:]
    return final_img

size = (200,200)
final_img = generateImage(size)

fig, [ax1,ax2] = plt.subplots(1,2)
ax1.set_title('Image from experiments')
ax1.imshow(img, 'gray')
ax2.set_title('Generated image')
ax2.imshow(new_img, 'gray')
ax2.imshow(final_img, 'gray')
        
        
        
        
    