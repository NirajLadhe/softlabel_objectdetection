# What kind of dataset COCO-LVIS is?


The dataset contains real-world images with objects annotated by human raters. Each annotation describes one object in one image. The annotation contains the object class, bounding box, segmentation polygon, image ID, and rater ID. 
The task was not to train a detector directly, but to integrate this dataset into the existing soft-label object detection pipeline.

Annotation files of COCO-LVIS was in **JSON format**, so in order to run it through pipeline we needed to normalize it to CSV format.

The JSON format is:

```
{
  "area": 4500.6,
  "category_id": 951,
  "bbox": [432.34, 195.75, 84.22, 72.27],
  "segmentation": [[...]],
  "image_id": 305309,
  "id": 12245,
  "iscrowd": 0,
  "rater_id": "r1"
}
```

Where:
```
area                  Area of the annotated object

category_id           Object class ID

bbox	              Bounding box in [x, y, width, height] format

segmentation          Polygon mask points for the object

image_id              ID of the image where the object appears

iscrowd               COCO-style crowd flag

rater_id              Annotator/rater who labeled the object
```

For this assignment we focused on **Bounding boxes, category labels, image ID's and rater ID's**

We needed to map the images, mapping is important because the soft label pipeline needs to know which rater created which box, so it can compare annotations from different raters and calculate uncertainty.
```
image-id -> image file
bbox -> object box coordinates
category_id -> object class
rater_id -> annotator identity
```

# Dataset Format

The dataset has three important sections:
```
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

The `images` section contains image metadata:
```
{
  "license": 6,
  "width": 640,
  "file_name": "000000336232.jpg",
  "height": 427,
  "id": 336232,
  "rater_list": ["r1", "r2"]
}
```

The `annotations` section contains object annotations:
```
{
  "area": 238.39,
  "category_id": 1161,
  "bbox": [442.95, 132.95, 26.96, 22.13],
  "image_id": 398203,
  "id": 12246,
  "iscrowd": 0,
  "rater_id": "r1"
}
```

and the `categories` section mapped class ID's to class names.

The bounding box format was:
```

# What needed to be done in order pipeline to work?

Checked all the files available.
Inspected the structure of **JSON** files.

Created a profilling script: `/home/niraj/softlabel_objectdetection/src/softlabel_od/softlabel_generation/core/preprocessing/profile_lvis.py`

The profilling checked:
```
number of images
number of annotations
number of categories
number of raters
missing rater IDs
missing image references
missing category references
invalid bounding boxes
rater distribution
category distribution
```

The generated `.md` files are stored at:
`/home/niraj/softlabel_objectdetection/src/softlabel_od/softlabel_generation/core/preprocessing/docs`
```


We needed to pre-process the COCO-LVIS in order to normalize it
COCO-Lvis Uses:
```
[x, y, width, height]
```
But the internal pipeline expected normalized coordinates. which needed to be converted:
```
x_min = x
y_min = y
x_max = x + width
y_max = y + height
```
The Normalized them:
```
x_min_norm = x_min / image_width
y_min_norm = y_min / image_height
x_max_norm = x_max / image_width
y_max_norm = y_max / image_height
```
The pre-processing script is at:
`/home/niraj/softlabel_objectdetection/src/softlabel_od/softlabel_generation/core/preprocessing/preprocess_lvis.py`

The preprocessing handled the following:
```
load raw LVIS JSON
read images, annotations, and categories
map image_id to image metadata
map category_id to category name
convert bbox from [x, y, w, h] to [x_min, y_min, x_max, y_max]
normalize bbox coordinates
rename rater_id to rad_id where needed
export normalized CSV
export COCO-style JSON for the pipeline
save image lookup file for visualization
```

Changes needed to be done in Step 1 and Step 2 for LVIS dataset in `run_pipeline.py`.
Changed relaibility output naming which was hardcoded for `vindrcxr` in `src/softlabel_od/softlabel_generation/core/reliability/calculate_reliability.py`
The semantic mapping was for `vindrcxr`, it needed to be changed as well in `src/softlabel_od/softlabel_generation/utils/semantic_mapping.py`

There was bounding box scaling issue in reliability.
The issue was that width and height needed to be fetched for every annotation before converting normalized boxes back to pixel coordinates.
The changes are done at :
`src/softlabel_od/softlabel_generation/core/reliability/kalos_reliability.py`

#### The command to run the full pipeline is:
```
python -m softlabel_od.scripts.run_pipeline \
  --dataset lvis \
  --raw-lvis-json /media/klee_ro/public_datasets/coco_lvis/lvis_standardized_lvis_val.json \
  --lvis-output-dir outputs/lvis \
  --gmm-mode 1d \
  --clustering-method iou \
  --iou-threshold 0.5 \
  --iou-reliability 0.5 \
  --reliability-method ahc \
  2>&1 | tee outputs/lvis/pipeline_lvis_full.log

```

### Exported the first 5 soft-labels outputs at:
`/home/niraj/Assignment2/docs/lvis_first5_softlabels.md`

### Visualized high-disagreement examples at:
`/home/niraj/Assignment2/outputs/lvis/visualizations`

#### Image 1
![High disagreement image 1](../outputs/lvis/Images/lvis_disagreement_1_lvis_274687.png)

#### Image 2
![High disagreement image 2](../outputs/lvis/Images/lvis_disagreement_2_lvis_71226.png)

#### Image 3
![High disagreement image 3](../outputs/lvis/Images/lvis_disagreement_3_lvis_317433.png)

#### Image 4
![High disagreement image 4](../outputs/lvis/Images/lvis_disagreement_4_lvis_18575.png)

#### Image 5
![High disagreement image 5](../outputs/lvis/Images/lvis_disagreement_5_lvis_266400.png)



