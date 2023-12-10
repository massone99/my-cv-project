def load(root, photosDir, labelsFile):
    """
    Load photos and corresponding labels from the given directory.

    Args:
        root (str): Root directory path.
        photosDir (str): Directory path where photos are stored.
        labelsFile (str): File path of the labels file.

    Returns:
        list: List of photos and their corresponding labels.
    """
    photos = FileSystem.__read_files(os.path.join(root, photosDir))

    # Read labels file
    path = os.path.join(root, labelsFile)
    with open(path, "r") as f:
        data = json.load(f)

    # Extract the list of images and annotations
    result = []
    annotations = [(image['-name'], image['box']['-xtl'], image['box']['-ytl'],
                    image['box']['-xbr'], image['box']['-ybr']) for image in data['annotations']['image']]
    for photo in photos:
        found = [ann for ann in annotations if Path(ann[0]).stem == photo['filename']]
        assert len(found) == 1
        result.append([photo['img'], [1, # only class (logo identifier 1)
                                        float(found[0][1]), float(found[0][2]),
                                        float(found[0][3]), float(found[0][4])]])

    return result


