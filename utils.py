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
    
    iterations = (total_files_t // batch_size) + 1 if total_files_t % batch_size != 0 else (total_files_t // batch_size)
    idx = 0
    for ii in range(iterations):
        new_idx = idx + batch_size if idx+batch_size < total_files_t else total_files_t
        inputs = []
        labels = []
        
        for jj in range(idx, new_idx):
            img = plt.imread(training_path + filenames_t[jj])
            img = cv2.resize(img, (512, 512), cv2.INTER_CUBIC)
            inputs.append(img)
            
            img = cv2.imread(segmented_path + filenames_s[jj])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), cv2.INTER_NEAREST)
            mask = create_mask(img, color_map)
            labels.append(mask)
         
        inputs = np.stack(inputs, axis=2)
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)
        
        labels = torch.tensor(labels)
        
        yield inputs, labels
        
        idx = new_idx
