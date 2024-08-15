class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def xyxy(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def from_skimage_region(region):
        return BoundingBox(
            xmin=region.bbox[1],
            ymin=region.bbox[0],
            xmax=region.bbox[3],
            ymax=region.bbox[2]
        )
    
    def __str__(self):
        return f'BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})'