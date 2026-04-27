## Task 1 

(LedDet1) niraj@itten:~$ hostname
itten
(LedDet1) niraj@itten:~$ nvidia-smi
Sun Apr 26 01:29:59 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-16GB           Off |   00000000:61:00.0 Off |                    0 |
| N/A   33C    P0             25W /  250W |       4MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-PCIE-16GB           Off |   00000000:DB:00.0 Off |                  Off |
| N/A   32C    P0             24W /  250W |       4MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1634      G   /usr/lib/xorg/Xorg                              4MiB |
|    1   N/A  N/A      1634      G   /usr/lib/xorg/Xorg                              4MiB |
+-----------------------------------------------------------------------------------------+
(LedDet1) niraj@itten:~$ pip list
Package                  Version      Editable project location
------------------------ ------------ -------------------------------------
aiohappyeyeballs         2.6.1
aiohttp                  3.13.5
aiosignal                1.4.0
async-timeout            5.0.1
attrs                    26.1.0
contourpy                1.3.2
cuda-bindings            13.2.0
cuda-pathfinder          1.5.3
cuda-toolkit             13.0.2
cycler                   0.12.1
docstring_parser         0.18.0
filelock                 3.29.0
fonttools                4.62.1
frozenlist               1.8.0
fsspec                   2026.3.0
idna                     3.12
ImageIO                  2.37.3
importlib_resources      7.1.0
iterative-stratification 0.1.9
Jinja2                   3.1.6
joblib                   1.5.3
jsonargparse             4.48.0
kalos                    0.7.3
kiwisolver               1.5.0
krippendorff             0.8.2
lazy-loader              0.5
lightning-utilities      0.15.3
MarkupSafe               3.0.3
matplotlib               3.10.8
mpmath                   1.3.0
multidict                6.7.1
networkx                 3.4.2
numpy                    2.2.6
nvidia-cublas            13.1.0.3
nvidia-cuda-cupti        13.0.85
nvidia-cuda-nvrtc        13.0.88
nvidia-cuda-runtime      13.0.96
nvidia-cudnn-cu13        9.19.0.56
nvidia-cufft             12.0.0.61
nvidia-cufile            1.15.1.6
nvidia-curand            10.4.0.35
nvidia-cusolver          12.0.4.66
nvidia-cusparse          12.6.3.3
nvidia-cusparselt-cu13   0.8.0
nvidia-nccl-cu13         2.28.9
nvidia-nvjitlink         13.0.88
nvidia-nvshmem-cu13      3.4.5
nvidia-nvtx              13.0.85
packaging                26.1
pandas                   2.3.3
pillow                   12.2.0
pip                      22.0.2
propcache                0.4.1
pycocotools              2.0.11
pylibmgm                 1.1.2
pyparsing                3.3.2
python-dateutil          2.9.0.post0
pytorch-lightning        2.6.1
pytz                     2026.1.post1
PyYAML                   6.0.3
scikit-image             0.25.2
scikit-learn             1.7.2
scipy                    1.15.3
seaborn                  0.13.2
setuptools               59.6.0
shapely                  2.1.2
six                      1.17.0
softlabel_od             0.1.0        /home/niraj/softlabel_objectdetection
sympy                    1.14.0
threadpoolctl            3.6.0
tifffile                 2025.5.10
torch                    2.11.0
torchmetrics             1.9.0
torchvision              0.26.0
tqdm                     4.67.3
triton                   3.6.0
typeshed_client          2.10.0
typing_extensions        4.15.0
tzdata                   2026.1
yacs                     0.1.8
yarl                     1.23.0


### Git cloning and branch successfully created 

  niraj@itten:~/softlabel_objectdetection$ git branch --show-current
niraj
niraj@itten:~/softlabel_objectdetection$ git log --oneline -3
4453653 (HEAD -> niraj, origin/niraj, origin/main, origin/Sahil_onboarding, origin/HEAD, main) anchor_setup_to_config
26a1e89 Merge branch 'lightning' into 'main'
11930b1 resolve_package_and_paths
niraj@itten:~/softlabel_objectdetection$ 

## Task 2




### Q. What is Soft-Label?
1. Soft labels are probability-based labels that represent uncertainty, disagreement, and confidence instead of pretending there is one perfectly correct answer. 


Instead of giving a certain answer, a soft label gives a probability of possible answers, in this case it is a disease. Soft labels preserve uncertainty.
With bounding box, 
a hard label will look like :
```
        box = [x, y, width, height]
```

A soft label will look like
```
Box mean =  [x, y, width, height]
Box std = [uncertainty_x, uncertainty_y, uncertainty_w, uncertainty_h]
```

## Q. A detailed breakdown of the steps used to generate the soft labels. Describe how the raw annotation CSV is processed and transformed into soft labels.
### 1. Step 0 :
When we run the file `enrich_csv.py`, it adds width and height to each annotations row by looking up the image metadata or by reading the image file because dataset might not always contain the image dimensions. 


### Step 1:
#### Normalize the annotation CSV


 Basically coordinate standardization


#### Why normalize?
Images can be in different sizes, so if a box says `x_min = 200` that does not indicate the same coordinate in different images. 
So to normalize we convert from absolute position to percentage position
         
For example, suppose the image size is:
```
width = 1000 pixels
height = 800 pixels
```
and the disease box is:
```
x_min = 200
y_min = 100
x_max = 600
y_max = 500
```
After normalization:
```
x_min = 200 / 1000 = 0.20
y_min = 100 / 800  = 0.125
x_max = 600 / 1000 = 0.60
y_max = 500 / 800  = 0.625
```
So the normalized box becomes:
```
[0.20, 0.125, 0.60, 0.625]
```
This means:
```
left edge of box   = 20% from the left side of the image
top edge of box    = 12.5% from the top side
right edge of box  = 60% from the left side
bottom edge of box = 62.5% from the top side
```
That is much better than raw pixels because now the box means the same thing regardless of image resolution.
The normalized coordinate system works like this:
```
x = 0 means left edge of image
x = 1 means right edge of image
```
```
y = 0 means top edge of image
y = 1 means bottom edge of image
```
So a full-image box is:
```
[0, 0, 1, 1]
```
which means:
```
start at top-left corner
end at bottom-right corner
cover the full image
```

For “`No finding`”, it does not say the disease covers the whole image instead it says “**there is not disease box, but it we still need a placeholder box so the pipeline can process this row**”


### Step 2: Convert normalized CSV to COCO-style JSON 
This step reorganizes the dataset into a proper object-detection format so that clustering, reliability scoring, and soft-label generation can be done cleanly. 
Converting data from a table format to an object detection dataset format.


The normalized CSV is easy for humans to read, but later object-detection and reliability-processing code usually expects data in a structured JSON format, similar to the COCO dataset format. 


#### What is happening in this step?


The code converts the CSV into a structured JSON with three sections: 

```
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
} 
```

The images section stores each unique image only once. 
For example: 

```
{
  "id": "img_001",
  "file_name": "img_001.png",
  "width": 1000,
  "height": 800,
  "rad_ids": ["R1", "R2", "R3"]
} 
```

This indicates the `image id - “img_001”`, it’s size and which radiologists annotated it.


The rad_ids list is important later because the pipeline needs to know not only who marked a finding, but also who looked at the image and did not mark that finding. That is how the pipeline can create soft negative votes like NoFindingInCluster. 


The categories section stores the class names for VinDe-CXR, these are pathology labels.
For example:
```

        Id = 3
        Name = cardiomegaly
```

And 

```
        Id = 14
        Name = no finding
```





The annotations section stores the actual bounding boxes. Each row from the CSV becomes one annotation object, as long as the box is valid. 


A CSV row like this:
```
image_id = img_001
rad_id = R1
class_id = 3
class_name = Cardiomegaly
x_min = 0.20
y_min = 0.125
x_max = 0.60
y_max = 0.625
```
is converted into something like:
```
{
 "id": 1,
          "image_id": "img_001",
          "category_id": 3,
          "rater_id": "R1",
          "bbox": [0.20, 0.125, 0.40, 0.50],
          "area": 0.20
}
```
#### The important change is the bounding box format.

In the CSV, the box is stored as:
```
x_min, y_min, x_max, y_max
```
This means:
```
x_min = left edge of the box
y_min = top edge of the box
x_max = right edge of the box
y_max = bottom edge of the box
```
But COCO-style format stores the box as:
```
[x, y, width, height]
So the code converts:
x = x_min
y = y_min
bbox_width = x_max - x_min
bbox_height = y_max - y_min
```
Using the example:
```
x_min = 0.20
y_min = 0.125
x_max = 0.60
y_max = 0.625
```
we get:
```
bbox_width = 0.60 - 0.20 = 0.40
bbox_height = 0.625 - 0.125 = 0.50
```
So the COCO-style box becomes:
```
[0.20, 0.125, 0.40, 0.50]
```
This means:
```
start the box at x = 0.20
start the box at y = 0.125
box width = 0.40 of the image width
box height = 0.50 of the image height
```
The area is then calculated as:
```
area = bbox_width × bbox_height
```
So in this example:
```
area = 0.40 × 0.50 = 0.20
```
Because the coordinates are normalized, the area is also normalized. An area of 0.20 means the bounding box covers around 20% of the image area, not 20 pixels.


The code also checks for the invalid boxes, a box is invalid if it has zero or negative width or height.

For No Finding, the annotation is handled differently. Since there is no actual pathology location, the pipeline stores:
```
{
           "category_id": 14,
           "bbox": [0, 0, 1, 1],
           "area": 1
}
```
This means the box starts at the top-left corner and covers the full image:
```
x = 0
y = 0
width = 1
height = 1
```
Again, this is not saying the whole image is diseased. It is only a placeholder. The real meaning is:
```
this radiologist found no abnormality in this image
```
The reason the placeholder is used is that the JSON format expects every annotation to have a bounding box. Instead of making special exceptions everywhere for No Finding, the pipeline gives it a dummy full-image box so it can keep the same structure for every annotation.

### Step 3: Calculate annotator reliability 
        The pipeline loads the COCO-style JSON, The rater_list is especially important because it tells the code which radiologists annotated each image. 
        The code collects all annotators, and it compares these raters against each other. This creates pairwise agreement scores
        The bounding boxes are compared using IoU

```
        IoU = Intersection over union


        IoU = overlap area / total combined area

```
If two raters draw almost same box, the IoU is high. Two boxes are considered to match if they overlap by at least 50% according to IoU


This is necessary because two radiologists may select the same disease class but draw boxes in totally different places. That should not be counted as a strong agreement. 


**K∝LOS / Vision Alpha calculates the agreement**


The job of the matching algorithm is to decide which annotation from one rater corresponds to which annotation from another rater. This is important because an image can contain multiple findings.


Pairwise agreement matrix is created which looks like this:
 ```         
          R1      R2      R3
R1        -      0.80    0.60
R2      0.80      -      0.50
R3      0.60    0.50      -
```

The core idea is : reliability is estimated from how consistently one annotator agrees with the other annotators.

 One reliability scores is calculated per annotator.




The pipeline then converts the pairwise scores into one score per radiologist.
For each radiologist, it calculates the average of all pairwise agreement scores involving that radiologist.
Example:
```
R1-R2 = 0.80
R1-R3 = 0.60
```
Then:
```
R1 Avg K-alpha = (0.80 + 0.60) / 2 = 0.70
```
For R2:
```
R2-R1 = 0.80
R2-R3 = 0.50
```
So:
```
R2 Avg K-alpha = (0.80 + 0.50) / 2 = 0.65
```
For R3:
```
R3-R1 = 0.60
R3-R2 = 0.50
```
So:
```
R3 Avg K-alpha = (0.60 + 0.50) / 2 = 0.55
```
The output would roughly mean:
```
R1 is most reliable.
R2 is second.
R3 is least reliable.
```
Not because R3 is “bad” in an absolute sense, but because R3 agrees less with the group compared with the others.


The reliability saves a CSV like:
```
Annotator
Avg K-alpha
K-alpha Std Dev
Participation (Assigned Images)
Consistency Factor (K-alpha) 
```
`Annotator` is the radiologist ID.

`Avg K-alpha` is the main reliability score.
It is the average agreement between that annotator and all other annotators.


Higher means:
```
this annotator agrees more consistently with the rest of the group
```
Lower means:
```
this annotator disagrees more often in class choice, box position, or both.
```
This is the column actually used later for soft-label weighting.


`K-alpha Std Dev` This measures how stable the annotator’s agreement is across other raters.
Example:
```
R1-R2 = 0.90
R1-R3 = 0.30
```
Average may be okay, but the variation is large. That means R1 agrees strongly with one rater but poorly with another.
Low standard deviation means the annotator behaves more consistently across different comparisons.

`Participation (Assigned Images)` 
Participation indicates how many valid pairwise reliability comparisons contributed to that annotator’s score.


`Consistency Factor (K-alpha)`
This is calculated as:
```
Consistency Factor = 1 - K-alpha Std Dev
```
So if the standard deviation is small, consistency is high.
Example:
```
K-alpha Std Dev = 0.10
Consistency Factor = 0.90
If the standard deviation is large:
K-alpha Std Dev = 0.40
Consistency Factor = 0.60
```
But again, important: in your downstream soft-label generation, the weights are based on `Avg K-alpha`, not directly on this consistency factor.

Reliability scores are normalized into weights
Later, the soft-label generation script loads the reliability CSV and normalizes the `Avg K-alpha values`.

Suppose the CSV has:
```
R1 Avg K-alpha = 0.70
R2 Avg K-alpha = 0.65
R3 Avg K-alpha = 0.55
```
The total is:
```
0.70 + 0.65 + 0.55 = 1.90
```
Then the normalized weights become:
```
R1 weight = 0.70 / 1.90 = 0.368
R2 weight = 0.65 / 1.90 = 0.342
R3 weight = 0.55 / 1.90 = 0.289
```
So the final weights are:
```
R1 = 0.368
R2 = 0.342
R3 = 0.289
```
These weights sum to 1.
This means R1 has slightly more influence than R3 when generating the final soft labels.

How the weights affair final class probabilities
If multiple radiologists annotate the same cluster:
```
R1: Cardiomegaly
R2: Cardiomegaly
R3: Aortic Enlargement
```

And we treat all raters equally, the class probability will become:
```
        {
            "Cardiomegaly": 0.67,
            "Aortic Enlargement": 0.33
       }
```

But if we use reliability weights:
```        
R1 weight = 0.368
R2 weight = 0.342
R3 weight = 0.289
```
Then :
``` 
        Cardiomegaly = R1 + R2 = 0.368 + 0.342 = 0.710
        Aortic Enlargement = R3 = 0.289
```
So the final class probability becomes :
``` 
        {
            "Cardiomegaly": 0.710,
            "Aortic Enlargement": 0.289
        }

```
This is the main reason reliability is calculated. The pipeline does not blindly count votes. It counts weighted votes. 


#### How the weights affect bounding box location
Suppose three radiologists draw slightly different boxes:
R1 box x = 0.20
R2 box x = 0.22
R3 box x = 0.35
If all were equal, the average x-position would be:
(0.20 + 0.22 + 0.35) / 3 = 0.256
But with reliability weights:
R1 = 0.368
R2 = 0.342
R3 = 0.289
the weighted average becomes:
x_mean = 0.20×0.368 + 0.22×0.342 + 0.35×0.289 
 = 0.250
So the final box is pulled more toward the more reliable raters.
That is the point: the final soft label is not just the average annotation; it is a reliability-weighted annotation.




This does not mean that pipeline knows “True correct radiologist”
It only estimates reliability based on agreement with the group.
The annotator reliability is estimated based on the agreements he has with other raters, assuming that annotators who agree more consistently with others provide more stable labels. 


### Step 4: Calculate class-specific priors 


        A prior is a backup uncertainty estimate. It tells the model, “Even if this image has only one annotation, do not pretend the box is perfectly certain. For this disease type, radiologists usually vary by this much.” 


        Why this step is needed
Suppose only one radiologist marks a finding:
R1: Cardiomegaly box = [0.20, 0.30, 0.40, 0.25]
There is only one box. If you calculate variance from one value, the variance is:
0
That would mean:
The box location is perfectly certain.
That is obviously false. It only means there was no second annotation to compare against.
This is the main problem Step 4 solves. It prevents the pipeline from becoming stupidly overconfident just because there is only one annotation.


For 1D mode, it calculates uncertainty separately for : 
        X
Y
Width
height
Prior is not a single number, it is four separate uncertainty values.


Before calculating priors, the code groups annotations that probably describe the same finding. 
The default clustering method is : 
        IoU clustering with iou threshold = 0.5
So if boxes overlap enough they are treated as the same findings, if they are far apart, they are treated as different findings.


Then all clusters are grouped by class, after clustering each cluster is assigned a class.
         1) if all annotators agree, cluster class is assigned with agreement
2) if annotators disagree, cluster class is assigned using majority vote.


#### Only multi-annoted clusters are useful
The code only uses clusters with at least two annotations, because a single box cannot show disagreement.
Including single-annotation clusters would artificially reduce variance and make the priors too confident.


#### How variance is calculated inside one cluster?
        Suppose three radiologists annotate the same Cardiomegaly finding:
R1 bbox = [0.20, 0.30, 0.40, 0.25]
R2 bbox = [0.22, 0.31, 0.38, 0.27]
R3 bbox = [0.18, 0.29, 0.42, 0.24]
Each box is in:
[x, y, width, height]
The code separates the coordinates:
x values      = [0.20, 0.22, 0.18]
y values      = [0.30, 0.31, 0.29]
width values  = [0.40, 0.38, 0.42]
height values = [0.25, 0.27, 0.24]
Then it calculates variance separately:
variance of x
variance of y
variance of width
variance of height
So one cluster produces four variance numbers:
{
                   "x_var": 0.00027,
                   "y_var": 0.00007,
                   "w_var": 0.00027,
                   "h_var": 0.00016
}
Interpretation:
x_var tells how much radiologists disagreed about the left position.
y_var tells how much they disagreed about the top position.
w_var tells how much they disagreed about box width.
h_var tells how much they disagreed about box height.
Low variance means radiologists drew very similar boxes.
High variance means radiologists disagreed more.




        Code averages variance across all clusters of that class
        
        The class needs at least 5 multi-annoted clusters before the pipeline trusts its class-specific prior.This is necessary because if a class has only 1 or 2 examples, its priors will be noisy and unreliable. If we use that it will give fake precision.


        Global prior 
        If a class does not have enough multi-annoted examples, the pipeline uses a global prior.


        The global prior is calculated by pooling multi-annoted clusters from all classes.
        All their variances are averaged together
        So if a rare disease X does not have enough samples, the pipeline says:
                I do not know the typical uncertainty for this exact disease,
so I will use the average uncertainty across all diseases.


This is not perfect, but is better than assuming zero uncertainty.


        How alpha controls the strength of the prior 
                
                We can think of ∝ as weight of prior.
                      adjusted_variance =
(n × instance_variance + alpha × prior_variance) / (n + alpha)


                Where: 
                        n = number of annotations in the current cluster
instance_variance = disagreement inside the current image cluster
prior_variance = learned typical disagreement for that class
alpha = strength of the prior






                        With alpha = 1.0
                
                The prior behaves like one extra pseudo-annotation


        If current cluster has many annotations, the real observed disagreements dominates. If the cluster has only one or two annotations, prior has stronger influence.




                
#### How annotator weights can affect variance
The prior calculation itself uses unweighted variance when computing class priors from clusters.
But later, when adjusting variance for a specific cluster, the code can use annotator reliability weights.
So if reliable raters draw similar boxes and one less reliable rater draws a box far away, the weighted variance is less affected by the unreliable rater.




The prior file contains three main parts:


                {
 "metadata": {
                   "alpha": 1.0,
                   "min_samples_per_class": 5,
                   "total_classes": 10,
                   "has_global_fallback": true
 },
  "class_priors": {
                    "Cardiomegaly": {
                      "x_var": 0.00030,
                      "y_var": 0.00025,
                      "w_var": 0.00050,
                      "h_var": 0.00045,
                      "n_samples": 42
                    }
                 },
                 "global_prior": {
                   "x_var": 0.00040,
                   "y_var": 0.00035,
                   "w_var": 0.00055,
                   "h_var": 0.00050,
                   "n_samples": 250
  }
}




### Step 5: Generate final soft labels 


        Step 5 generates the final soft labels by combining clustered annotations, annotator reliability weights, and class-specific spatial priors. For each image, all pathology annotations are grouped into finding clusters using the selected overlap-based clustering method, usually IoU clustering with a threshold of 0.5. Each cluster represents annotations that probably refer to the same underlying finding. A participant constraint is applied so that each radiologist can contribute at most one annotation to a cluster, preventing duplicate boxes from giving one annotator excessive influence.
For every cluster, the class probability distribution is calculated using reliability-weighted votes from the annotators. If multiple annotators assign different class names to the same cluster, the final label preserves this disagreement as soft class probabilities instead of forcing a single hard class. Annotators who reviewed the image but did not mark the cluster contribute to the special class NoFindingInCluster, representing soft negative evidence.
The spatial component of the soft label is calculated by taking a reliability-weighted mean of the bounding box coordinates [x, y, width, height]. The uncertainty is stored as standard deviations for each coordinate. These standard deviations are obtained from prior-adjusted variances, which combine the observed disagreement inside the current cluster with the class-specific uncertainty priors calculated in Step 4. This prevents single-annotation clusters from receiving zero uncertainty. The final output is a JSON file where each finding contains the image ID, original annotations, weighted class probabilities, mean bounding box, spatial uncertainty, prior used, and the number of annotations in the cluster. This final label represents both semantic disagreement and spatial uncertainty, making it more informative than a traditional hard annotation.
Final soft-label structure : 


{
  "image_id": "some_image_id",
  "image_width": 1024,
  "image_height": 1024,
  "finding_id": 0,
  "original_annotations": [
    {
      "rater_id": "R1",
      "bbox": [0.22, 0.31, 0.15, 0.18],
      "class_name": "Cardiomegaly",
      "weight": 0.36
    }
  ],
  "spatial": {
    "mean": [0.23, 0.30, 0.16, 0.19],
    "std": [0.02, 0.03, 0.01, 0.02]
  },
  "prior_used": "Cardiomegaly",
  "class_probs": {
    "Cardiomegaly": 0.72,
    "NoFindingInCluster": 0.28
  },
  "num_annotations_in_cluster": 2
}




image_id  
which X-ray image the finding belongs to


image_width and image_height 
These store the original image size. Even though the box coordinates are normalized, storing the original dimensions is useful if you later want to convert boxes back to pixel coordinates. 


finding_id 
This is the ID of the finding cluster inside that image.


original_annotations 
This stores the original radiologist annotations that contributed to the cluster.
It is useful because you can inspect where the final soft label came from.
This makes softlabel traceable.


#### spatial.mean
This is the final representative bounding box.


#### Spatial.std
This is the uncertainty around the mean box.
Higher value means more disagreement or uncertainty.
Lower value means stronger agreement.


#### prior_used
This tells which prior was used to stabilize the variance estimate.
This matters because a class-specific prior is usually more meaningful than a global fallback


#### class_probs
This is the final semantic soft label.
It is not a single class. It is probability like distribution over possible classes.




num_annotations_in_cluster 


This tells how many actual annotation boxes were grouped into this finding cluster.
        If this number is high, the finding has stronger multi-rater support.
        If this number is low, especially 1, the label depends more heavily on priors and soft negative evidence.
#### Important limitation
Do not oversell the output as “ground truth.” It is not true ground truth.
It is a probabilistic consensus label based on:
radiologist annotations
box overlap clustering
annotator reliability
class-specific uncertainty priors
So the final soft label is better than a hard majority vote, but it is still only as good as the original annotations and the assumptions in the pipeline.
#### The main assumptions are:
1. Overlapping boxes refer to the same finding.
2. Annotators who agree more with others are more reliable.
3. Class-specific disagreement patterns are reusable for future annotations.
4. NoFindingInCluster represents meaningful negative evidence.
These assumptions are reasonable, but not perfect.





### Output of first five `image_id` from `1DGMM_softlabel`
```
{
    "image_id": "50a418190bc3fb1ef1633bf9678929b3",
    "image_width": 2332,
    "image_height": 2580,
    "finding_id": 0,
    "original_annotations": [
      {
        "rater_id": "R11",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.05165310154622143
      },
      {
        "rater_id": "R15",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.049299716110238236
      },
      {
        "rater_id": "R16",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.05193631698550881
      }
    ],
    "spatial": {
      "mean": [
        0.0,
        0.0,
        1.0,
        1.0
      ],
      "std": [
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    "prior_used": "no_finding",
    "class_probs": {
      "No finding": 1.0
    },
    "num_annotations_in_cluster": 3
  },
  {
    "image_id": "21a10246a5ec7af151081d0cd6d65dc9",
    "image_width": 2954,
    "image_height": 3159,
    "finding_id": 0,
    "original_annotations": [
      {
        "rater_id": "R7",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.06743224744937523
      },
      {
        "rater_id": "R13",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.04926599998651355
      },
      {
        "rater_id": "R2",
        "bbox": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "class_name": "No finding",
        "weight": 0.06317727263531965
      }
    ],
    "spatial": {
      "mean": [
        0.0,
        0.0,
        1.0,
        1.0
      ],
      "std": [
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    "prior_used": "no_finding",
    "class_probs": {
      "No finding": 1.0
    },
    "num_annotations_in_cluster": 3
  },
  {
    "image_id": "9a5094b2563a1ef3ff50dc5c7ff71345",
    "image_width": 2080,
    "image_height": 2336,
    "finding_id": 0,
    "original_annotations": [
      {
        "rater_id": "R10",
        "bbox": [
          0.3322115384615384,
          0.5886130136986302,
          0.4625,
          0.1952054794520548
        ],
        "class_name": "Cardiomegaly",
        "weight": 0.057904070884778516
      },
      {
        "rater_id": "R9",
        "bbox": [
          0.3326923076923077,
          0.5886130136986302,
          0.46394230769230765,
          0.18150684931506844
        ],
        "class_name": "Cardiomegaly",
        "weight": 0.05774223349090001
      },
      {
        "rater_id": "R8",
        "bbox": [
          0.33125,
          0.5620719178082192,
          0.4697115384615385,
          0.19263698630136983
        ],
        "class_name": "Cardiomegaly",
        "weight": 0.057984989581717765
      }
    ],
    "spatial": {
      "mean": [
        0.33205031113979033,
        0.5797494895399583,
        0.4653879762320867,
        0.18979214912987707
      ],
      "std": [
        0.003381983457613891,
        0.013507649696224356,
        0.005754685123550898,
        0.01060417470083314
      ]
    },
    "prior_used": "Cardiomegaly",
    "class_probs": {
      "Cardiomegaly": 1.0
    },
    "num_annotations_in_cluster": 3
  },
  {
    "image_id": "9a5094b2563a1ef3ff50dc5c7ff71345",
    "image_width": 2080,
    "image_height": 2336,
    "finding_id": 1,
    "original_annotations": [
      {
        "rater_id": "R9",
        "bbox": [
          0.8600961538461539,
          0.740154109589041,
          0.0413461538461537,
          0.11258561643835618
        ],
        "class_name": "Pleural effusion",
        "weight": 0.05774223349090001
      }
    ],
    "spatial": {
      "mean": [
        0.8600961538461539,
        0.740154109589041,
        0.0413461538461537,
        0.1125856164383562
      ],
      "std": [
        0.013225157064629308,
        0.022749627569867887,
        0.019335375154695515,
        0.031721066179022685
      ]
    },
    "prior_used": "Pleural effusion",
    "class_probs": {
      "Pleural effusion": 0.33255660413996657,
      "NoFindingInCluster": 0.6674433958600333
    },
    "num_annotations_in_cluster": 1
  },
  {
    "image_id": "9a5094b2563a1ef3ff50dc5c7ff71345",
    "image_width": 2080,
    "image_height": 2336,
    "finding_id": 2,
    "original_annotations": [
      {
        "rater_id": "R9",
        "bbox": [
          0.8600961538461539,
          0.740154109589041,
          0.0413461538461537,
          0.11258561643835618
        ],
        "class_name": "Pleural thickening",
        "weight": 0.05774223349090001
      }
    ],
    "spatial": {
      "mean": [
        0.8600961538461539,
        0.740154109589041,
        0.0413461538461537,
        0.1125856164383562
      ],
      "std": [
        0.008034555833763515,
        0.008384283205378573,
        0.012185505620399167,
        0.013346961615908214
      ]
    },
    "prior_used": "Pleural thickening",
    "class_probs": {
      "Pleural thickening": 0.33255660413996657,
      "NoFindingInCluster": 0.6674433958600333
    },
    "num_annotations_in_cluster": 1
  },
```
  
### Main hyperparameters and settings used

From `run_pipeline.py`, the default generation settings are:
```
gmm_mode = 1d
input_csv = outputs/vindrcxr/train_enriched.csv
normalized_csv = outputs/vindrcxr/train_enriched_normalized.csv
coco_json = outputs/vindrcxr/train_vindrcxr_all.json

no_finding_id = 14

reliability_backend = kalos
reliability_method = shm
iou_reliability = 0.5

clustering_method = iou
iou_threshold = 0.5
iomin_threshold = 0.5
center_ratio_threshold = 0.5

semantic_map = vindrcxr

prior_alpha = 1.0
min_samples = 5
max_images = None
```
The generated output is:
```
outputs/vindrcxr/1d_gaussian_softlabels/
1DGMM_softLabels_iou0_5_with_priors.json
```
