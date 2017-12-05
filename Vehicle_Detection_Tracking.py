
# coding: utf-8

# In[1]:


##Implement Some Importatnt functions to be used in the project flow


# In[2]:


# Important imports
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog

import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
imageio.plugins.ffmpeg.download()


# In[3]:


# Important notes when reading images in this project

'''
png, mpimg -> 0 - 1 
png, cv2   -> 0 - 255
jpg, mpimg -> 0 - 255
jpg, cv2   -> 0 - 255
'''


# ## 1- Draw Boxes Function

# In[4]:


# Draw boxes using cv2 library given 2 opposite points
def draw_boxes(img, bboxes, color, thick):
    # make a copy from the input image
    draw_img = np.copy(img)
    # draw the bounding box which has the input opposite points in shape of ((x1,y1),(x2,y2))
    for bbox in bboxes:
        # draw the rectangle using cv2.rectangle with the input color of shape (R,G,B)
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    
    return draw_img


# In[5]:


# Test draw_boxes Function
test_image = mpimg.imread("test_images/test1.jpg")

test_bboxes = [((800,500),(950,400))]
test_result = draw_boxes(test_image,test_bboxes, color=(255,0,0), thick=8)
plt.imshow(test_result)
plt.show()


# 
# # 2- Features Extraction 

# ## 2a) Color Hitograms Features

# In[6]:


# Extract features from the color histogram
def color_hist_features(img, nbins, bins_range):
    # Calclate the histograms for each channel seperately
    chan1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range) 
    chan2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    chan3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generate bins centers
    bins_edges = chan1_hist[1]
    bins_centers = (bins_edges[1:] + bins_edges[0:len(bins_edges)-1])/2
    # Concatenate all features together
    color_hist_features = np.concatenate((chan1_hist[0], chan2_hist[1], chan3_hist[0]))
    
    # return the histogram features which is the most important one from this function
    # However, the otehr histograms and bins centers will be needed to be visualized in testing this function
    return color_hist_features, chan1_hist, chan2_hist, chan3_hist, bins_centers


# In[7]:


# Test Color histogram features extraction
test_image = mpimg.imread("test_images/test1.jpg")

test_features, test_ch1, test_ch2, test_ch3, test_centers = color_hist_features(test_image, 
                                                                                nbins=32, 
                                                                                bins_range=(0,256))

fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.bar(test_centers, test_ch1[0])
plt.xlim(0, 256)
plt.title('ch1 Histogram')
plt.subplot(132)
plt.bar(test_centers, test_ch2[0])
plt.xlim(0, 256)
plt.title('ch2 Histogram')
plt.subplot(133)
plt.bar(test_centers, test_ch3[0])
plt.xlim(0, 256)
plt.title('ch3 Histogram')
fig.tight_layout()
plt.show()


# ## 2b) Color Spatial Bining Features

# In[8]:


# Extract features from the Color spatial Bining
def bin_spatial(img, color_space, size):
    # convert the image into the color space sent into the function
    if color_space != "RGB":
        if color_space == "HSV":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "HLS":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "LUV":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == "YUV":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == "YCrCb":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == "GRAY":
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        feature_img = np.copy(img)
    
    # flatten the features extarcted from the image after resizing
    bin_spatial_features = cv2.resize(feature_img, size).ravel()
    
    # return these features
    return bin_spatial_features, feature_img


# In[9]:


# Test Color spatial Bining 
test_image = mpimg.imread("test_images/test1.jpg")
print(test_image.shape)

test_features, test_img_converted = bin_spatial(test_image, color_space="YCrCb", size=(8,8))
plt.plot(test_features)
plt.show()


# ## 2c) Oriented Gradient Histogram features (Hog)

# In[10]:


# Extract features of the Histogram Oriented Gradient
def get_hog_features(img, orient, pix_per_cell, cell_per_block, transform_sqrt_flag, vis_flag, feature_vector_flag):
    # Note that the img here should be 2D (grayscale)
    # check the visualization flag if it is true or not to plot the output of hog functionality 
    if vis_flag == True:
        # apply hog with visualizing the output of hog functionality
        hog_features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell,pix_per_cell),
                                      cells_per_block=(cell_per_block,cell_per_block),
                                      transform_sqrt=transform_sqrt_flag, 
                                      visualise=vis_flag, feature_vector=feature_vector_flag)
        
        return hog_features, hog_image
        
    if vis_flag == False:
        # apply hog without visualizing the output of hog functionality
        hog_features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell,pix_per_cell),
                           cells_per_block=(cell_per_block,cell_per_block),
                           transform_sqrt=transform_sqrt_flag, 
                           visualise=vis_flag, feature_vector=feature_vector_flag)
        
        return hog_features
    


# In[11]:


# Test extraction of hog features and visulaize
test_image = mpimg.imread("test_images/test1.jpg")

# Note that the image should be 2D (grayscale) 
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
test_features, test_result_img = get_hog_features(test_gray, orient=12, pix_per_cell=4, 
                                                  cell_per_block=2, transform_sqrt_flag=True, 
                                                  vis_flag=True, feature_vector_flag=True)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(test_image, cmap='gray')
plt.title('Example')
plt.subplot(122)
plt.imshow(test_result_img, cmap='gray')
plt.title('HOG Visualization')
plt.show()


# # 3- Combine Features (Color_hist, bin_spatial) with (hog)

# In[12]:


# Extract all of the previous features from list of images 
def extract_features(imgs, cspace, spatial_size, hist_nbins, hist_range,  
                     orient, pix_per_cell, cell_per_block, transform_sqrt_flag, vis_flag, feature_vector_flag,
                     hog_channel, extract_spatial_flag, extract_color_hist_flag, extract_hog_flag, cv2read=False):
    # Create empty list for appending the extracted features
    features = []
    # make an iteration to apply the extraction over img by img

    for img in imgs:
        # create local features for every image to preserve them after finishing all images
        image_features = []
        # read the img
        if cv2read == True:
            image_read = cv2.imread(img)
        else:
            image_read = img
            
        #converted_image = image_read
        
        # Apply bin spatial features extraction
        bin_features, converted_image = bin_spatial(image_read, color_space=cspace, size=spatial_size)
        
        # Apply color hist features extraction
        col_features,_,_,_,_ = color_hist_features(converted_image, nbins=hist_nbins, bins_range=hist_range)
        
        # Apply hog features extraction
        if hog_channel == "ALL":
            hog_features = []
            # Apply hog features extraction over each channel in the image
            for channel in range(converted_image.shape[2]):
                hog_features.append(get_hog_features(img=converted_image[:,:,channel], orient=orient, 
                                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                                     transform_sqrt_flag=transform_sqrt_flag, 
                                                     vis_flag=vis_flag, feature_vector_flag=feature_vector_flag))
            hog_features = np.ravel(hog_features)
        else:
            # Apply hog features extraction over the given channel in the image
            hog_features = get_hog_features(img=converted_image[:,:,hog_channel], orient=orient, 
                                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                                     transform_sqrt_flag=transform_sqrt_flag, 
                                                     vis_flag=vis_flag, feature_vector_flag=feature_vector_flag)
        
        # Append all of the features in one list
        if extract_spatial_flag == True:
            image_features.append(bin_features)
        if extract_color_hist_flag == True:
            image_features.append(col_features)
        if extract_hog_flag == True:
            image_features.append(hog_features)
        
        #print(image_features)
        
        # Appned all of the features in (features) list after concatenate all of the previous features
        features.append(np.concatenate(image_features))
        
    # return all of these features in a feature vector
    return features


# In[13]:


def extract_features_One_image(img, cspace, spatial_size, hist_nbins, hist_range,  
                     orient, pix_per_cell, cell_per_block, transform_sqrt_flag, vis_flag, feature_vector_flag,
                     hog_channel, extract_spatial_flag, extract_color_hist_flag, extract_hog_flag, cv2read=False):
    
    # Create empty list for appending the extracted features
    features = []
    # create local features for every image to preserve them after finishing all images
    image_features = []
    # read the img
    if cv2read == True:
        image_read = cv2.imread(img)
    else:
        image_read = img
    
    #converted_image = image_read

    # Apply bin spatial features extraction
    bin_features, converted_image = bin_spatial(image_read, color_space=cspace, size=spatial_size)

    # Apply color hist features extraction
    col_features,_,_,_,_ = color_hist_features(converted_image, nbins=hist_nbins, bins_range=hist_range)

    # Apply hog features extraction
    if hog_channel == "ALL":
        hog_features = []
        # Apply hog features extraction over each channel in the image
        for channel in range(converted_image.shape[2]):
            hog_features.append(get_hog_features(img=converted_image[:,:,channel], orient=orient, 
                                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                                 transform_sqrt_flag=transform_sqrt_flag, 
                                                 vis_flag=vis_flag, feature_vector_flag=feature_vector_flag))
        hog_features = np.ravel(hog_features)
    else:
        # Apply hog features extraction over the given channel in the image
        hog_features = get_hog_features(img=converted_image[:,:,hog_channel], orient=orient, 
                                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                                 transform_sqrt_flag=transform_sqrt_flag, 
                                                 vis_flag=vis_flag, feature_vector_flag=feature_vector_flag)

    # Append all of the features in one list
    if extract_spatial_flag == True:
        image_features.append(bin_features)
    if extract_color_hist_flag == True:
        image_features.append(col_features)
    if extract_hog_flag == True:
        image_features.append(hog_features)

    # Appned all of the features in (features) list after concatenate all of the previous features
    features.append(np.concatenate(image_features))

    # return all of these features in a feature vector
    return features


# # 4- HeatMap, apply threhold, draw labeled bboxes Functions

# ## 4a) HeatMap Function

# In[14]:


# add heatmap using the bounding boxes list given as an input to the function
def add_heat(heatmap, bbox_list):
    # note that heamap input here is zeros of the shape of the image or one channel only in the image
    # iterate through the bboxlist
    for bbox in bbox_list:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        
    # return the heatmap
    return heatmap


# ## 4b) Apply threshold on the heatmap created

# In[15]:


# apply threshold value over the heatmap created
def apply_threshold(heatmap, threshold):
    # values below the given threshold will be equal to 0
    heatmap[heatmap <= threshold] = 0
    
    return heatmap


# ## 4c) draw labeled bboxes 

# In[16]:


# draw the bounding box rectangle on the image given the labels 
def draw_labels_bboxes(img, labels):
    # note that labels will be come from scipy.ndimage.measurements
    #iterate through the whole detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# # 5- Build Classifier (Normalize, Train, Test, Accuracy calculation)

# ## 5a) Extract features of both Cars, NotCars data

# In[17]:


# Read all images paths for cars and notcars
Cars_images   = glob.glob("vehicles/*/*.png")
noCars_images = glob.glob("non-vehicles/*/*.png") 

# save images in these lists
cars = []
notcars = []


for car_image in Cars_images:
    cars.append(car_image)
    
for notcar_image in noCars_images:
    notcars.append(notcar_image)

# sample_size = 1000
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

# parameters need tweak 
color_space = 'YCrCb'       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = "ALL"         # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)     # Spatial binning dimensions
spatial_range = (0, 200)    # Spatial range
hist_bins = 32              # Number of histogram bins
spatial_transform = True   # Spatial Transform sqrt
spatial_feat = True         # Spatial features on or off
visualize = False           # Visualization flag
feature_vector = True      # Feature Vector flag
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off
y_start_stop = [400, 680] # Min and max in y to search in slide_window()

car_features = extract_features(imgs=cars, cspace=color_space, spatial_size=spatial_size, hist_nbins=hist_bins, 
                                hist_range=spatial_range, orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, transform_sqrt_flag=spatial_transform, 
                                vis_flag=visualize, feature_vector_flag=feature_vector, hog_channel=hog_channel, 
                                extract_spatial_flag=spatial_feat, extract_color_hist_flag=hist_feat, 
                                extract_hog_flag=hog_feat, cv2read=True)

Notcar_features = extract_features(imgs=notcars, cspace=color_space, spatial_size=spatial_size, hist_nbins=hist_bins, 
                                hist_range=spatial_range, orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, transform_sqrt_flag=spatial_transform, 
                                vis_flag=visualize, feature_vector_flag=feature_vector, hog_channel=hog_channel, 
                                extract_spatial_flag=spatial_feat, extract_color_hist_flag=hist_feat, 
                                extract_hog_flag=hog_feat, cv2read=True)


# ## 5b) Normalize, Labels, SVC-Classifier, Train, Accuracy Calculation

# In[18]:


# Combine features of cars and notcars together
X = np.vstack((car_features, Notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(Notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC() # C=5.0, gamma='auto', kernel='rbf'
#svc = SVC(C=5.0,kernel='rbf')

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')


# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


# ## 5c) Save the parameters needed after that in a pickle file

# In[19]:


# Save the parameters in a pickle in order to be easily accessed
pickle_file = "Classifier.p"
print("Saving the data in a pickle file.....")

with open(pickle_file, "wb") as p_file:
    pickle.dump({"X_Scaler": X_scaler,
                 "svc":svc,
                 "cspace": color_space,
                 "orient": orient,
                 "pix_per_cell": pix_per_cell,
                 "cell_per_block": cell_per_block,
                 "hog_channel":hog_channel,
                 "spatial_size": spatial_size,
                 "hist_bins":hist_bins,
                 "spatial_range": spatial_range,
                 "spatial_transform":spatial_transform,
                 "visualize":visualize,
                 "feature_vector": feature_vector,
                 "spatial_feat": spatial_feat,
                 "hist_feat": hist_feat,
                 "hog_feat": hog_feat,
                 "y_start_stop": y_start_stop }, p_file, pickle.HIGHEST_PROTOCOL)


# # 6- Sliding Windows method 

# In[20]:


# Sliding window search to get the windows 
def slide_window(img, x_start_stop, y_start_stop, xy_window, xy_overlap):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# In[21]:


# test the Sliding window function
test_image = mpimg.imread("test_images/test1.jpg")

windows = slide_window(test_image, x_start_stop=[0, 1300], y_start_stop=[400, 750], xy_window=(96, 96), xy_overlap=(0.5, 0.5))
test_result = draw_boxes(test_image, windows, (255,0,0), 8)
plt.imshow(test_result)
plt.show()


# # 7- Search Windows

# ## 7a) Search Window and prediction for the input image

# In[22]:


# This function will take an image as an input and list of windows to search in them 
def search_windows(img, windows, classifier, scaler, cspace, spatial_size, hist_nbins, hist_range,  
                     orient, pix_per_cell, cell_per_block, transform_sqrt_flag, vis_flag, feature_vector_flag,
                     hog_channel, extract_spatial_flag, extract_color_hist_flag, extract_hog_flag):
    
    
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
    
        # 4) Extract features for that window using single_img_features()
        features = extract_features_One_image(img=test_img, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, 
                                hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, transform_sqrt_flag=transform_sqrt_flag, 
                                vis_flag=vis_flag, feature_vector_flag=feature_vector_flag, hog_channel=hog_channel, 
                                extract_spatial_flag=extract_spatial_flag, 
                                extract_color_hist_flag=extract_color_hist_flag, extract_hog_flag=extract_hog_flag)
        
        #print(features.min())
        #print(features.max())
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        
        # 6) Predict using your classifier
        prediction = classifier.predict(test_features)
        #print("pred: ", prediction)
        
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# ## 7b) Load the pickle that conatins the needed parameters

# In[23]:


# read the data saved previously in the pickle file
pickle_file_name = "Classifier.p"

with open(pickle_file_name, "rb") as f:
    pickle_data = pickle.load(f)
    
#     # X_scaler
#     param_X_scaler = pickle_data["X_Scaler"]
#     param_svc = pickle_data["svc"]
#     param_color_space = pickle_data["cspace"]
#     param_orient = pickle_data["orient"]
#     param_pix_per_cell = pickle_data["pix_per_cell"]
#     param_cell_per_block = pickle_data["cell_per_block"]
#     param_hog_channel = pickle_data["hog_channel"]
#     param_spatial_size = pickle_data["spatial_size"]
#     param_hist_bins = pickle_data["hist_bins"]
#     param_spatial_transform = pickle_data["spatial_transform"]
#     param_visualize = pickle_data["visualize"]
#     param_feature_vector = pickle_data["feature_vector"]
#     param_hist_feat = pickle_data["hist_feat"]
#     param_hog_feat = pickle_data["hog_feat"]
#     param_y_start_stop = pickle_data["y_start_stop"]
    

print("Saved parameters is loaded..")
  


# # 8) Apply the full pipeline

# ## 8a) Find Cars in the image based on the loaded data

# In[24]:


# This function car find cars in an image based on the saved data in the pickle file
def find_cars(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #plt.show()
    # Extract parameters from the pickle file
    pickle_file_name = "Classifier.p"

    with open(pickle_file_name, "rb") as f:
        parameters = pickle.load(f)
    
    # copy from the image passed to the function
    draw_image = np.copy(image)
    y_start_stop = parameters["y_start_stop"]

    # get windows from slide_window function with multiscale sliding window
    windows = slide_window(image, x_start_stop=[None, 1300], y_start_stop=[400,600],
                            xy_window=(64, 64), xy_overlap=(0.9, 0.9))
    
    windows += slide_window(image, x_start_stop=[None, 1300], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.9, 0.9))

    windows += slide_window(image, x_start_stop=[None, 1300], y_start_stop=y_start_stop,
                            xy_window=(128, 128), xy_overlap=(0.9, 0.9))
    

    # search in the windows we have to select the best windows 
    hot_windows = search_windows(img=image, windows=windows, classifier=parameters["svc"],
                                 scaler=parameters["X_Scaler"], cspace=parameters["cspace"], 
                                 spatial_size=parameters["spatial_size"], hist_nbins=parameters["hist_bins"],
                                 hist_range=parameters["spatial_range"],orient=parameters["orient"],
                                 pix_per_cell=parameters["pix_per_cell"],
                                 cell_per_block=parameters["cell_per_block"],
                                 transform_sqrt_flag=parameters["spatial_transform"],
                                 vis_flag=parameters["visualize"],
                                 feature_vector_flag=parameters["feature_vector"], 
                                 hog_channel=parameters["hog_channel"],
                                 extract_spatial_flag=parameters["spatial_feat"], 
                                 extract_color_hist_flag=parameters["hist_feat"],
                                 extract_hog_flag=parameters["hog_feat"])
    
    
    # draw boxes over the given image
    window_image = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=8)
    
    #plt.imshow(window_image)
    #plt.show()
    
    # Create a zeros_like the image given in order to be passed over the function of heatmap
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    
    #print(heat.max())

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 15)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    #plt.imshow(heatmap)
    #plt.show()

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image = draw_labels_bboxes(np.copy(draw_image), labels)

    # plt.close("all")
    #
#     fig = plt.figure()
#     plt.figure(figsize=(20,10))
#     #
#     plt.subplot(133)
#     plt.imshow(draw_image)
#     plt.title('Car Positions')
#     plt.subplot(132)
#     plt.imshow(heatmap, cmap='hot')
#     plt.title('Heat Map')
#     plt.subplot(131)
#     plt.imshow(window_image)
#     plt.title('Windows')
    # # fig.tight_layout()
    # # mng = plt.get_current_fig_manager()
    #
    # # mng.full_screen_toggle()
    # # plt.pause(0.05)
    #
    # # plt.imshow(window_img)
    #plt.show()
    
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)
    
    return draw_image


# ## 8b) Test the pieline using the test images in the folder we have

# In[25]:


# group_of_images = glob.glob("test_images/*.jpg")

# for image in group_of_images:
#     read_test_image_to_pipeline = cv2.imread(image)

#     print(pickle_data)
#     test_result_image_from_pipeline = find_cars(read_test_image_to_pipeline)

#     plt.imshow(cv2.cvtColor(test_result_image_from_pipeline, cv2.COLOR_BGR2RGB))
#     plt.show()


# # 9) Apply the Pipeline on the Project Video

# In[26]:


# # Extract frames from the test video

# project_output_video = "test_output_video.mp4"
# clip = VideoFileClip("test_video.mp4")
# output_video = clip.fl_image(find_cars) #     NOTE: this function expects color images!!
# get_ipython().magic('time output_video.write_videofile(project_output_video, audio=False)')


# In[27]:


# group_of_images = glob.glob("test_images/*.jpg")

# for image in group_of_images:
#     read_test_image_to_pipeline = cv2.imread(image)

#     print(pickle_data)
#     read_test_image_to_pipeline=cv2.cvtColor(read_test_image_to_pipeline, cv2.COLOR_RGB2BGR)
#     test_result_image_from_pipeline = find_cars(read_test_image_to_pipeline)

#     #plt.imshow(cv2.cvtColor(test_result_image_from_pipeline, cv2.COLOR_BGR2RGB))
#     plt.imshow(test_result_image_from_pipeline)
#     plt.show()


# In[28]:


# Extract frames from the Project video

# project_output_video = "project_output_video.mp4"
# clip = VideoFileClip("project_video.mp4")
# output_video = clip.fl_image(find_cars) #     NOTE: this function expects color images!!
# get_ipython().magic('time output_video.write_videofile(project_output_video, audio=False)')


# In[29]:


# project_output_video = "test_output_video.mp4"
# clip = VideoFileClip("test_video.mp4")
# output_video = clip.fl_image(find_cars).subclip(0,5) #     NOTE: this function expects color images!!
# clip.write_videofile(project_output_video)
# # %time clip.write_videofile(project_output_video, audio=False)


# In[30]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
imageio.plugins.ffmpeg.download()


# In[31]:


# project_output_video = "test.mp4"
# clip = VideoFileClip("test_video.mp4")
# output_video = clip.fl_image(find_cars).subclip(0,1) #  .subclip(0,35)   NOTE: this function expects color images!!
# %time output_video.write_videofile(project_output_video, audio=False)


# In[32]:


# project_output_video = "part1.mp4"
# clip1 = VideoFileClip("project_video.mp4")
# output_video = clip1.fl_image(find_cars).subclip(0,10) #  NOTE: this function expects color images!!
# %time output_video.write_videofile(project_output_video, audio=False)


# In[33]:


project_output_video_2 = "Output.mp4"
clip2 = VideoFileClip("project_video.mp4")
output_video_2 = clip2.fl_image(find_cars) #  NOTE: this function expects color images!!
get_ipython().magic('time output_video_2.write_videofile(project_output_video_2, audio=False)')


# In[ ]:




