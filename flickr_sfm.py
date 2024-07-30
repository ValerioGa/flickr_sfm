import sys
import os

sys.path.append("Hierarchical-Localization")

import argparse
import shutil
import time
from pathlib import Path
import subprocess
import numpy as np
from tqdm import tqdm
import random
import pycolmap
import h5py
from typing import Any, Dict, List, Optional
import time

from hloc import logger
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval, pairs_from_exhaustive
from hloc.triangulation import (
    import_features,
    import_matches,
)
from hloc.utils.database import COLMAPDatabase


def create_test_set(train_folder, test_folder, num_files=100):
    files = os.listdir(train_folder)
    files_to_move = random.sample(files, num_files)
    
    logger.info(f"Selecting {num_files} images for test set")
    for file in tqdm(files_to_move):
        percorso_file_sorgente = os.path.join(train_folder, file)
        percorso_file_destinazione = os.path.join(test_folder, file)
        shutil.move(percorso_file_sorgente, percorso_file_destinazione)

def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def remove_images(self, image_name):

    id = self.execute(
        "SELECT image_id FROM images WHERE name=(?)",
        (image_name,),
    )
    self.execute(
        "DELETE FROM images WHERE name=(?)",
        (image_name,),
    )
    
    return id.fetchall()

def remove_keypoints(self, image_id):
    self.execute(
        "DELETE FROM keypoints WHERE image_id=(?)",
        (image_id,),
    )

MAX_IMAGE_ID = 2**31 - 1
def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2

def get_matches_ids(self):
    ids = self.execute("SELECT pair_id FROM matches")

    return ids.fetchall()

def remove_match(self, match_id):
    self.execute(
        "DELETE FROM matches WHERE pair_id=(?)",
        (match_id,),
    )
    self.execute(
        "DELETE FROM two_view_geometries WHERE pair_id=(?)",
        (match_id,),
    )

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


if __name__ == "__main__":
    # PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_lat", type=float, required=True)
    parser.add_argument("--max_lat", type=float, required=True)
    parser.add_argument("--min_lon", type=float, required=True)
    parser.add_argument("--max_lon", type=float, required=True)
    parser.add_argument("--images_folder", type=str, required=True, help="Folder where to save flickr images")
    parser.add_argument("--test_mode", type=bool, default=False, help="if True it just adds the test set images and does the evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output dir")
    parser.add_argument("--retrieval_n", type=int, default=20, help="number of retrieval images")
    parser.add_argument("--retrieval_confs", type=str, default="netvlad", choices=["netvlad", "salad", "cosplace", "no_retrieval"])
    parser.add_argument("--feature_confs", type=str, default="superpoint_aachen", choices=["superpoint_aachen", "superpoint_inloc", "superpoint_max", "sift", "disk"])
    parser.add_argument("--matching_confs", type=str, default="superglue", choices=["superglue", "NN-superpoint", "disk+lightglue", "NN-ratio"])
    parser.add_argument("--min_retrieval_score", type=float, default=0, help="minimum retrieval score to be accepted while making pairs")
    parser.add_argument("--skip_topk_retrieval", type=int, default=0, help="number of top retrieval matches to discard")
    parser.add_argument("--max_iter_ba", type=int, default=25, help="colmap mapper option")
    parser.add_argument("--ba_images_ratio", type=float, default=1.2, help="colmap mapper option")
    parser.add_argument("--ba_points_ratio", type=float, default=1.2, help="colmap mapper option")
    args = parser.parse_args()


    os.makedirs(args.images_folder, exist_ok=True)
    train_images = Path(os.path.join(args.images_folder, 'train'))
    test_images = Path(os.path.join(args.images_folder, 'test'))
    outputs = Path(os.path.join(args.output_dir, args.images_folder.split('/')[-1],
                               f"{args.retrieval_confs}{args.retrieval_n}_min{args.min_retrieval_score}_skip{args.skip_topk_retrieval}+{args.feature_confs}+{args.matching_confs}"))
    sfm_pairs = outputs / 'pairs.txt'
    sfm_dir = outputs / 'sfm'

    if args.test_mode is False:

        #If image folders do not already exist, download them from Flickr 
        if len(os.listdir(args.images_folder))==0:
            os.makedirs(train_images)
            os.makedirs(test_images)

            subprocess.run(["python3", "IMCvenv/sfm/download_flickr.py", "--min_lat", str(args.min_lat), "--max_lat", str(args.max_lat),
                            "--min_lon", str(args.min_lon), "--max_lon", str(args.max_lon), 
                            "--output_folder", str(args.images_folder),])
            create_test_set(train_images, test_images)

        start_time = time.time()

        feature_conf = extract_features.confs[str(args.feature_confs)]
        retrieval_conf = extract_features.confs[str(args.retrieval_confs)]
        matcher_conf = match_features.confs[str(args.matching_confs)]

        os.makedirs(outputs, exist_ok=True)

        #Find pairs through retrieval
        if args.retrieval_confs == 'no_retrieval':
            pairs_from_exhaustive.main(sfm_pairs, image_list=os.listdir(train_images))
        else:
            retrieval_path = extract_features.main(retrieval_conf, train_images, outputs)
            pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=args.retrieval_n, min_score=args.min_retrieval_score, skip_topk=args.skip_topk_retrieval)

        #Extract and match local features
        feature_path = extract_features.main(feature_conf, train_images, outputs)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
        
        #COLMAP reconstruction
        mapper_options = {'multiple_models': False, 'ba_refine_principal_point': True, 'ba_refine_extra_params': True, 
                        'ba_global_max_num_iterations': args.max_iter_ba, 'ba_global_images_ratio': args.ba_images_ratio, 'ba_global_points_ratio': args.ba_points_ratio}
        model = reconstruction.main(sfm_dir, train_images, sfm_pairs, feature_path, match_path, mapper_options=mapper_options)
        
        end_time = time.time()
        execution_time = end_time - start_time
    
    test_pairs = outputs / 'test-pairs.txt'

    feature_conf = extract_features.confs[args.feature_confs]
    matcher_conf = match_features.confs[args.matching_confs]
    retrieval_conf = extract_features.confs[args.retrieval_confs]

    retrieval_path = outputs / (retrieval_conf['output'] + ".h5")
    feature_path = outputs / (feature_conf['output'] + ".h5")
    match_path = outputs / (matcher_conf['output'] + ".h5")

    #Listing images registered in the reconstruction
    images_list = os.listdir(train_images)

    rec = pycolmap.Reconstruction(sfm_dir)
    reg_images = [img.name for img  in rec.images.values()]
    unreg_images = [img for img in images_list if img not in reg_images]

    num_reg_images = len(reg_images)

    #Removing unregistered images from global features file
    if args.retrieval_confs != 'no_retrieval':
        original_file = retrieval_path
        new_file = outputs / 'globfeats_new.h5'
        dataset_to_remove = reg_images

        with h5py.File(original_file, 'r') as src:
            with h5py.File(new_file, 'w') as dst:
                for name in src:
                    if name in dataset_to_remove:
                        src.copy(name, dst)

    query_list = sorted(os.listdir(test_images))

    logger.info("Processing test set images...")

    #Find test images pairs through retrieval
    if args.retrieval_confs == 'no_retrieval':
        pairs_from_exhaustive.main(test_pairs, image_list=query_list, ref_list=reg_images)
    else:
        retrieval_path_test = extract_features.main(retrieval_conf, test_images, image_list=query_list, feature_path=new_file, overwrite=False)
        pairs_from_retrieval.main(retrieval_path_test, test_pairs, num_matched=20, query_list=query_list)

    #Extract and match local features on test images
    extract_features.main(feature_conf, test_images, image_list=query_list, feature_path=feature_path, overwrite=False)
    match_features.main(matcher_conf, test_pairs, features=feature_path, matches=match_path, overwrite=False)


    #Updating COLMAP database
    database = sfm_dir / 'database.db'

    logger.info("Importing test images into COLMAP database...")
    import_images(test_images, database, camera_mode = pycolmap.CameraMode.AUTO, image_list=query_list)
    image_ids = get_image_ids(database)
    image_ids2 = {key: value for key, value in image_ids.items() if key in query_list}

    import_features(image_ids2, database, feature_path)
    import_matches(image_ids, database, test_pairs, match_path)

    #COLMAP match importer 
    subprocess.run(["colmap", "matches_importer", "--database_path", database, "--match_list_path", test_pairs])

    #Removing unregistered images
    logger.info("Removing unregistered images from COLMAP database...")
    removed_ids = []
    db = COLMAPDatabase.connect(database)

    logger.info("Removing images and keypoints...")
    for image in tqdm(unreg_images):

        id = remove_images(db, image)
        if id != []:
            removed_ids.append(int(id[0][0]))
            remove_keypoints(db, int(id[0][0]))


    logger.info("Removing matches...")
    matches_ids = get_matches_ids(db)
    for match in tqdm(matches_ids):
        pair_ids = pair_id_to_image_ids(match[0])
        
        if (pair_ids[0] in removed_ids) | (pair_ids[1] in removed_ids):
            remove_match(db, match[0])


    db.commit()
    db.close()

    logger.info("Registering test images into the reconstruction...")
    new_sfm_path = outputs / "sfm_with_test"
    os.makedirs(new_sfm_path)

    #COLMAP image registrator
    subprocess.run(["colmap", "image_registrator", "--database_path", database, "--input_path", sfm_dir, "--output_path", new_sfm_path])

    # Convert model to TXT format
    logger.info("Converting the new sfm model to txt format...")
    txt_model_path = new_sfm_path / 'txt'
    os.makedirs(txt_model_path)    
    subprocess.run(["colmap", "model_converter", "--input_path", new_sfm_path, "--output_path", txt_model_path, "--output_type", "TXT"])



    #create camera dict and points3D dict
    points3D_dict = {}
    with open(os.path.join(txt_model_path, 'points3D.txt'), "r") as points_file:
        for _ in range(3):
            next(points_file)
        
        for row in points_file:
            points3D_dict[row.split(' ')[0]] = row.split(' ')[1:4]

    cameras_dict = {}
    with open(os.path.join(txt_model_path, 'cameras.txt'), "r") as cameras_file:
        for _ in range(3):
            next(cameras_file)
        
        for row in cameras_file:
            cameras_dict[row.split(' ')[0]] = row.split(' ')[4:8]


    #Compute reprojection error
    rows = []
    with open(os.path.join(txt_model_path, 'images.txt'), "r") as images_file:
        for row in images_file:
            rows.append(row)

    logger.info("Computing reprojection error on test images...")
    mean_err = []
    point_err = []
    for i in np.arange(4, len(rows), 2):
        if rows[i].split(' ')[9].replace('\n', '') in query_list:
            image_id =  rows[i].split(' ')[0]
            camera_id = rows[i].split(' ')[8]
            
            Q = np.array(rows[i].split(' ')[1:5], float)
            R = qvec2rotmat(Q)
            T = np.array([rows[i].split(' ')[5:8]], float)
            K = np.array([[cameras_dict[camera_id][0], cameras_dict[camera_id][3].replace('\n', '') , cameras_dict[camera_id][1]],
                            [0, cameras_dict[camera_id][0], cameras_dict[camera_id][2]],
                            [0, 0, 1]], float)
            
            mat = np.hstack([R,T.T])
            cam_matrix = K@mat
            
            err = []
            count = 0
            for k in np.arange(2, len(rows[i+1].split(' ')), 3):
                if int(rows[i+1].split(' ')[k].replace('\n', '')) > 0:
                    count += 1
                    point_id = rows[i+1].split(' ')[k].replace('\n', '')
                    x_2d = rows[i+1].split(' ')[k-2]
                    y_2d = rows[i+1].split(' ')[k-1]

                    point3d = np.array(points3D_dict[point_id], float)
                    point3d = np.append(point3d, 1)
                    point_projection = cam_matrix@point3d.T
                    point_projection[0] = point_projection[0]/point_projection[2]
                    point_projection[1] = point_projection[1]/point_projection[2]
                    point_projection = np.delete(point_projection, -1)

                    p2d = np.array((x_2d, y_2d), float)

                    err.append(np.linalg.norm(p2d - point_projection))
                    point_err.append(np.linalg.norm(p2d - point_projection))
                
            mean_err.append([rows[i].split(' ')[9].replace('\n', ''), np.mean(err), count])


    #Write output file
    with open(os.path.join(outputs, "test_metrics.txt"), 'w') as file:
        if args.test_mode is False:
            file.write(f"Time for the reconstruction: {int(execution_time // 3600)}:{int((execution_time % 3600) // 60)}:{int(execution_time % 60)}\n")
        file.write(f"Registered images: {num_reg_images}\n")
        file.write(f"Registered test images: {len(mean_err)}/100\n")
        file.write(f"Mean reprojection error: {np.mean(point_err)}\n")
        for el in mean_err:
            file.write(str(el[0]) + ": error " + str(el[1]) + " on " + str(el[2]) + " points" + '\n')

    print(f"Registered {len(mean_err)}/100 test images")
    print(f"Mean reprojection error on test images: {np.mean(point_err)}")
