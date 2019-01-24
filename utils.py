import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_class_mask(img, color_map, is_normalized_img=True, is_normalized_map=False, show_masks=False):
    """
    Function to create C matrices from the segmented image, where each of the C matrices is for one class
    with all ones at the pixel positions where that class is present

    img = The segmented image

    color_map = A list with tuples that contains all the RGB values for each color that represents
                some class in that image

    is_normalized_img = Boolean - Whether the image is normalized or not
                        If normalized, then the image is multiplied with 255

    is_normalized_map = Boolean - Represents whether the color map is normalized or not, if so
                        then the color map values are multiplied with 255

    show_masks = Wherether to show the created masks or not
    """
    
    if is_normalized_img and (not is_normalized_map):
        img *= 255

    if is_normalized_map and (not is_normalized_img):
        img = img / 255
    
    mask = []
    hw_tuple = img.shape[:-1]
    for color in color_map:
        color_img = []
        for idx in range(3):
            color_img.append(np.ones(hw_tuple) * color[idx])

        color_img = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)

        mask.append(np.uint8((color_img == img).sum(axis = -1) == 3))

    return np.array(mask)


def loader(training_path, segmented_path, batch_size):
	"""
    The Loader to generate inputs and labels from the Image and Segmented Directory

    Arguments:

    training_path - str - Path to the directory that contains the training images

    segmented_path - str - Path to the directory that contains the segmented images

    batch_size - int - the batch size

    yields inputs and labels of the batch size
	"""

    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)
    
    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)
    
    assert(total_files_t == total_files_s)
    
    if str(batch_size).lower() == 'all':
        batch_size = total_files_s
    
    idx = 0
    while(1):
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
            
        
        inputs = []
        labels = []
        
        for jj in batch_idxs:
            img = plt.imread(training_path + filenames_t[jj])
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)
            
            img = Image.open(segmented_path + filenames_s[jj])
            img = np.array(img)
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)
         
        inputs = np.stack(inputs, axis=2)
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)
        
        labels = torch.tensor(labels)
        
        yield inputs, labels


def decode_segmap(image):
	Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colors = np.array([Sky, Building, Pole, Road_marking, Road, 
                              Pavement, Tree, SignSymbol, Fence, Car, 
                              Pedestrian, Bicyclist]).astype(np.uint8)

	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)

	for label in range(len(label_colors)):
		r[image == label] = label_colors[l, 0]
		g[image == label] = label_colors[l, 1]
		b[image == label] = label_colors[l, 2]

	rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
	rgb[:, :, 0] = b
	rgb[:, :, 1] = g
	rgb[:, :, 2] = r

	return rgb

def show3(img1, img2, img3, in_row=True):
	'''
	Helper function to show 3 images
	'''
	if not in_row:
		rc_tuple = (3, 1)

	figure = plt.figure(figsize=(20, 10))
	plt.subplot(*rc_tuple, 1)
	plt.title('Input Image')
	plt.axis('off')
	plt.imshow(img1)
	plt.subplot(*rc_tuple, 2)
	plt.title('Predicted Segmentation')
	plt.axis('off')
	plt.imshow(img2)
	plt.subplot(*rc_tuple, 3)
	plt.title('Ground Truth')
	plt.axis('off')
	plt.imshow(img3)
	plt.show()
