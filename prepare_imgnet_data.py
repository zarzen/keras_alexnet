from os import listdir, makedirs, rename
from os.path import isfile, join, exists

def get_files(folder):
    """
    Args:
        folder : path to folder
    """
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    return files

def get_idx(img_name):
    """
    Args:
        img_name: e.g. ILSVRC2011_val_00007144.JPEG
    Returns:
        integer part
    """
    name = img_name[:-5]
    parts = name.split('_')
    idx = int(parts[2])
    return idx

def get_ground_truth():
    """"""
    truth_path = './data/ILSVRC2011_validation_ground_truth.txt'
    with open(truth_path, 'r') as truth:
        labels = truth.readlines()
    labels = [l.strip() for l in labels]
    return labels

def categorize_img_with_label(labels, images):
    categories = {}
    for img in images:
        img_idx = get_idx(img)
        label = labels[img_idx-1]
        if label in categories:
            categories[label].append(img)
        else:
            categories[label] = [img]
    return categories


def move_img_to(imgs, label, source_folder, target_folder):
    """"""
    for img in imgs:
        if not exists(join(target_folder, label)):
            makedirs(join(target_folder, label))
        rename(join(source_folder, img),
               join(target_folder, label, img))


def split_data(categories, source_folder, val_folder, test_folder, train_folder):
    """"""
    for c in categories:
        imgs = categories[c]
        # train
        move_img_to(imgs[:40], c, source_folder, train_folder)

        # validation
        move_img_to(imgs[40:45], c, source_folder, val_folder)

        #test
        move_img_to(imgs[45:], c, source_folder, test_folder)


def main():
    labels = get_ground_truth()
    images = get_files('./data/images/')
    categories = categorize_img_with_label(labels, images)

    #move_img_to(img_with_label, './data/val', './data/train')
    split_data(categories, './data/images', './data/validation', './data/test', './data/train')


if __name__ == '__main__':
    main()
