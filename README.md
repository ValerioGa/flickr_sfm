# flickr_sfm
Structure from Motion on images downloaded from Flickr with evaluation

The code in this repository is meant to build 3D models using COLMAP incremental mapping, from images downloaded from Flickr. An evaluation metric is used to consistently compare different models. The focus is in particular at the image retrieval phase of the SfM pipeline, to extract image pairs on which local features should be matched.
The implementation exploits tools from `hloc` ([Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)) and adds new extractors and parameters. Additional extractors and matchers can be easily implemented if needed. 

The whole pipeline can be executed by running the script `flickr_sfm.py` with some arguments that will be explained in the following.

An output folder must be provided with the argument *output_dir*.



## Main pipeline
### Downloading images
The images are easily downloaded from Flickr by the script `download_flickr.py`. The images to download are selected based on their localization, so images with no geo-tag are discarded. The script downloads all the images with coordinates into the rectangle selected by the parameters *min_lat*, *min_lon*, *max_lat* and *max_lon*, which must be specified as `flickr_sfm.py` is called. A destination folder must be specified too. The images are saved in a sub-folder named "train", while 100 images are randomly selected and moved to another subfolder colled "test". These images will be used for the evaluation phase. It is necessary to select an area sufficiently large, so that it contains more than 100 images. If the number of test images is not suited for an experiment, it can be modified at line 32 of `flickr_sfm.py`, in the function *create_test_set*.

If a folder with the specified name, containing the train-test sub-folders, already exists, those images are used for the reconstruction and no other image is downloaded, even if the coordinates are different.

### Feature extraction and matching
Local descriptors (keypoints) are extracted on all the train images with the method specified in *feature_confs*. 

Then the images are paired using retrieval. An extractor specified in *retrieval_confs* is used to extract global descriptors and compute the similarity matrix. If it is set to the value *no_retrieval* images are paired exhaustively. This is suggested only for very small scenes, but in general retrieval helps to both reduce runtimes and increase the 3D model accuracy. Then each image is paired with the *N* most similar images, avoiding self matching. The pairs are written one for each line in the file `pairs.txt`. Other two parameters can be varied to get a fancier pair selection.

For each pair the keypoints of the two images are matched with the method specified in *matching_confs*.

Other extractors and matchers can be easily added if needed.

### 3D reconstruction
COLMAP is used to compute the reconstruction with the train images. Some parameters for the incremental mapper can be tuned easily using the corresponding command line arguments that will be explained in the following.

### Model evaluation
To evaluate the 3D model the proposed metric is the Mean Reprojection Error computed only on the points seen by at least one of the test images. This way the pool of images is always almost the same and so this metric can be used to compare consistently different models on the same scene.

Local features are extracted on the test images with the same method used for the main reconstruction. Pairs between a test image and a train image registered in the main model are selected with the same algorithm used before but with a fixed value of *N=20*. Pairs are saved in the file `test-pairs.txt`. Keypoints are then matched over these pairs using the same method as before. 

Unregistered train images and relative kaypoints and matches are removed from the COLMAP database, while test images and their keypoints and matches are added to it. The function *image_registrator* is called to register as many test images as possible in the reconstruction. Bundle adjustment is not performed, so that the test images are only localized, but do not affect the train points' and cameras' positions, that must be evaluated.

Each point seen by at least one test image is reprojected on it, using camera intrinsics and extrinsics. The *euclidean distance* between this reprojection and the corresponding keypoint originally extracted is computed and averaged over all the points.

The results are saved in the file `test_metrics.txt`, containing the time needed for the main reconstruction (on train images) expressed in [hours:minutes:seconds], the number of train images registered in the reconstruction, the number of test images registered and used for the evaluation and the Mean Reprojection Error computed as explained above. Then each of the following lines contain one of the registered test images with the error computed only on its points, and the number of keypoints seen, useful to search for some anomalies.

## Output folder

Outputs are saved in a directory with the name of the scene, specified in *images_folder*, inside the *output_dir*. Here a sub-directory with as a name the model parameters is created. For exampe a directory named `netvlad20_min0.0_skip0+superpoint_aachen+superglue` will contain the outputs of a model using NetVLAD for the retrieval, *N*=20, *min_score*=0, *k*=0, SuperPoint for the local descriptors extraction and SuperGlue for their matching.

output_dir/images_scene_name/model_name
  + sfm
    * images.bin
    * cameras.bin
    * points3D.bin
    * database.db
  + sfm_with_test
    * txt
      - images.txt
      - cameras.txt
      - points3D.txt
    * images.bin
    * cameras.bin
    * points3D.bin
  + local_features.h5
  + global_features.h5
  + matches.h5
  + global_features_new.h5
  + test_matches.h5
  + pairs.txt
  + test-pairs.txt
  + test_metrics.txt

The folder `sfm` will contain the largest model reconstructed with only the train images. `sfm_with_test` instead will contain the model where the test images have been registered, also converted in txt format. `test_metrics.txt` will contain the final results.

## test_mode

A test mode can be called with the relative command line parameter. In this case only the registration on test images and the evaluation will be performed. To use this mode images must be already saved and splitted between train and test. In addition the output folder must exist and contain the following files:

images_scene_name/model_name
  + sfm
    * models
    * images.bin
    * cameras.bin
    * points3D.bin
    * database.db
  + local_features.h5
  + global_features.h5
  + matches.h5
  + pairs.txt

## Command line parameters

The following parameters can be specified when `flickr_sfm.py` is called.

- **min_lat** (float, required=True): minimum latitude for the images to be downloaded from Flickr
- **max_lat** (float, required=True): maximum latitude for the images to be downloaded from Flickr
- **min_lon** (float, required=True): minimum longitude for the images to be downloaded from Flickr
- **max_lon** (float, required=True): maximum longitude for the images to be downloaded from Flickr
- **images_folder** (str, required=True): folder where to save the downloaded images
- **output_dir** (str, default="outputs"): folder where to save the outputs
- **test_mode** (bool, default=False): if True it only registers test images and performs the evaluation. In this case previous model is required
- **retrieval_confs** (str, default="netvlad"): extractor of global feature to use for image retrieval. Supported choices are "netvlad", "cosplace", "salad", "no_retrieval"
- **retrieval_n** (int, default=20): number of image pairs to be selected for each image. In other words each image is paired with the *N* most similar images in the train set
- **min_retrieval_score** (float, default=0): minimum similarity score an image pair should have to be accepted. It must be a value between 0 and 1. This can be usefull to discard wrong pairs and reduce runtimes
- **skip_topk_retrieval** (int, default=0): number of pairs with the higher similarity score to be discarded for each image. This can help to boost the model accuracy with some extractors, because pairs of images that share too similar points of view are less informative for the reconstruction.
- **feature_confs** (str, default="superpoint_aachen"): extractor of local descriptors (keypoints) to be used. Supported choices are "superpoint_aachen", "superpoint_inloc", "sift", "disk"
- **matching_confs** (str, default="superglue"): matcher to be used for keypoint matching across image pairs. Supported choices are "superglue", "NN-superpoint", "disk+lightglue", "NN-ratio". Be aware that not all the combinations of extractor-matcher can be used, since the two algorithms must support descriptors of the same size
- **max_iter_ba** (int, default=25): maximum number of global bundel adjustment iterations for COLMAP incremental mapper
- **ba_images_ratio** (float, default=1.2): the growth rate on the images number after which to perform global bundle adjustment
- **ba_points_ratio** (float, default=1.2): the growth rate on the points number after which to perform global bundle adjustment
