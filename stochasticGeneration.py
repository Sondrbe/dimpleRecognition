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
    plt.imshow(final_img)
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

# Define a performance measure function.
# As I am struggling a bit to understand the 2-point correlation function,
# I will define it using Monte Carlo sampling instead.
# This is slower, but easier:
"""

import numba
@numba.jit(nopython=True, locals={'distInd': numba.int32})
def twoPointLinealPathCorrelation(image):
    diffBuckets = int(0.5*min(image.shape[0], image.shape[1]))
    totalAttemptsOuter = 40000
    totalAttemptsInner = 5000
    samePhase = np.zeros(diffBuckets)
    totalExecuted = np.zeros(diffBuckets)    
    for i in range(totalAttemptsOuter):
        row1 = np.random.randint(low=0, high=image.shape[0])
        col1 = np.random.randint(low=0, high=image.shape[1])
        for i in range(totalAttemptsInner):
            row2 = np.random.randint(low=max(0,row1-30), high=min(row1+30, image.shape[0]))
            col2 = np.random.randint(low=max(0,col1-30), high=min(col1+30, image.shape[1]))
            dist = ((row1-row2)**2 + (col1-col2)**2)**0.5
            #if dist <= 0.5*min(image.shape[0], image.shape[1]):
            if dist <= 30:
                distInd = int(dist)
                totalExecuted[distInd] += 1
                if image[row1,col1] == 1:
                    if image[row2, col2] == 1:
                        samePhase[distInd] += 1  
    return samePhase/totalExecuted

performance = twoPointLinealPathCorrelation(final_img)
plt.plot(range(len(performance)), performance)
plt.plot(range(len(performance)), performance, 'k*')


























"""
Try different classifiers:
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#datasets = [T]
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    
scores = []    
classfs = []
for clf in classifiers:
    X_train, X_test, y_train, y_test = \
        train_test_split([x for x,_ in T], [y for _,y in T], random_state=42)
    clf.fit(X_train, y_train)
    classfs.append(clf)
    score = clf.score(X_test, y_test)
    scores.append(score)
    print('One done')

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1









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
"""

    






def generateImageFrom0s(size):
    import copy
    # Initialize a generated image to all 0's:
    new_img = np.zeros((size[0]+2*w, size[1]+2*h))
    
    c = 0.
    TOL = 0.1
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
    final_img = work_img[2*h:, 2*w:]
    return final_img
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
boundaryImg1 = generateImageFrom0s(boundaryImgSize1)
boundaryImg2 = generateImageFrom0s(boundaryImgSize2)
boundaryImg3 = generateImageFrom0s(boundaryImgSize3)
# Must initialize n rows of px's in each dimension. All outer rows are only 0's, 
# the inner is 6 generated images in correct size!
new_img[n-1,n-1:,n-1:] = boundaryImg2
new_img[n-1:,n-1:,n-1] = boundaryImg1
new_img[n-1:,n-1,n-1:] = boundaryImg3




"""
Now, the 3D distribution of particles are generated:
"""
def generate3dDistribution(new_img):
    c = 0.
    VVF_expected = None
    TOL = 0.04
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
                    pi_j = (p_ij_1 + p_ij_2 + p_ij_3)/3.
                    
                    p_ij_adjusted = p_ij + c*np.sqrt(p_ij*(1-p_ij))
                    x_ij = np.random.choice([1,0], p = [p_ij_adjusted, 1 - p_ij_adjusted])
                    work_img[px1, px2, px3] = x_ij
                    
                    # Tell the user how much calculations that has been performed:
                    if curr_px % int(tot_pxs/30) == 0:
                        print('Iteration '+str(iteration)+':', str(curr_px / tot_pxs) + '%')
        #final_img = new_img[2*n:, 2*n:, 2*n:]
        final_img = work_img[2*n:, 2*n:, 2*n:]
        VVF = np.sum(final_img) / np.prod(np.shape(final_img))
        if VVF_expected:
            if VVF < VVF_expected: c += 0.005
            else: c -= 0.005
        if ((VVF/VVF_expected)-1 <= TOL): break
        break
        print('An image is generated...')
    return final_img

final_3D_Image = generate3dDistribution(new_img)


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
        print('%f.2'%(px_count/(n1*n2*n3)*100), '%')
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
plotParticles3D(new_img)








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
        
        
        
        
    