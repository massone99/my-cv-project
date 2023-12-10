class YOLOReader:
    @staticmethod
    def read_yolo_format(file_path) -> list:
        """
        Read annotations in YOLO format from a file.

        Args:
            file_path (str): The path to the file containing the annotations.

        Returns:
            list: A list of tuples representing the annotations. Each tuple contains the class index, x center,
            y center, width, and height of an annotation.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            line = line.strip()
            class_index, x_center, y_center, width, height = map(float, line.split())
            annotations.append((class_index, x_center, y_center, width, height))
        return annotations
