import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
import statistics
import logging

from brainglobe_atlasapi import BrainGlobeAtlas

from .plotting import plot_probe_on_atlas
from .utils import cluster_color_map_for_labels, remap_labels, reshape


logger = logging.getLogger("lfploc")


def place_probe_on_atlas(df, probe_params, save_folder, norm, full_labels, AP, ML, DV, split_probes=False, custom_format="png"):

    if split_probes:
        probe_num = 2
    else:
        probe_num = 1
        
    full_labels_original = full_labels
    df_original = df
    DV_original = DV

    colors = False
    reshaped_labels_colored = False
    cluster_labels = False
    sorted_labels = False

    for x in range(probe_num):

        if split_probes:
            if x == 0:
                probe_params["n_shanks"] = 4
                full_labels = full_labels_original[384*x:384*(x+1)]
                ML = -0.75 + 0.375 #(half of probe width so ML is based on center of probe and not first shank)
                df = df_original.iloc[:384]
                DV = -4.2
            else:
                probe_params["n_shanks"] = 4
                full_labels = full_labels_original[384*x:384*(x+1)]
                #ML = -1.5 + 0.275 #(half of probe width so ML is based on center of probe and not first shank)
                df = df_original.iloc[384:]
                #df["X (um)"] -= 1000
                DV = DV_original

        # Bregma position in atlas (in micron)
        bregma_pos = [5400, 0, 0]

        # Load the Allen Mouse Brain Atlas
        #atlas_id = "kim_mouse_10um"
        atlas_id = "kim_mouse_isotropic_20um"
        atlas = BrainGlobeAtlas(atlas_id)

        # Load relabelled atlas where similar regions are plotted with the same color
        relabeled_atlas = np.load("./lib/relabeled_atlas.npy")
        rgb_map_relabelled = np.load("./lib/rgb_map.npy", allow_pickle=True).item()

        # get resolution from atlas_id
        # Extract resolution from atlas_id
        match = re.search(r'(\d+)\s*um', atlas_id)
        if match:
            resolution_um = int(match.group(1))
            print(f"Atlas resolution: {resolution_um} um")
        else:
            resolution_um = None
            print("Resolution not found in atlas_id.")

        # load structure information using json file
        json_path = os.path.join(atlas.root_dir, "structures.json")
        with open(json_path, "r") as f:
            structures = json.load(f)

        # Build a lookup: region_id -> RGB color
        id_to_rgb = {}
        all_rgb = []
        for region in structures:
            region_id = region['id']
            rgb = region.get('rgb_triplet', [255, 255, 255])  # fallback to white
            id_to_rgb[region_id] = rgb
            all_rgb.append(rgb)


        # reference image
        reference_image = atlas.reference

        # annotation image
        annotation_image = atlas.annotation
        annotation_shape = annotation_image.shape

        # a hemispheres image (value 1 in left hemisphere, 2 in right) can be generated
        hemispheres_image = atlas.hemispheres

        # get offset if using non-sinaps probes
        np_probe = True
        if np_probe:
            offset = probe_params["y_bottom (um)"] - 500
        else:
            offset = 0
            
        # Convert AP, ML, DV (mm) to µm
        AP_um = AP * 1000
        ML_um = ML * 1000
        DV_um = (DV * 1000) + offset

        # Convert AP, ML, DV (µm) to atlas voxel indices
        # AP: z axis, ML: x axis, DV: y axis
        slice_z = int((bregma_pos[0] - AP_um )/ resolution_um)
        slice_y = int((bregma_pos[1] + DV_um) / resolution_um)
        slice_x = int((bregma_pos[2] - ML_um) / resolution_um)

        n_shanks = probe_params['n_shanks']
        if n_shanks == 1: 
            selected_atlas = atlas.annotation
            selected_rgb_map = id_to_rgb
        else:
            selected_atlas = relabeled_atlas
            selected_rgb_map = rgb_map_relabelled


        # Prepare the annotation slice
        selected_annotated_image = selected_atlas[slice_z, :, :]
        selected_reference_image = reference_image[slice_z, :,:]

        # Map annotation IDs to RGB colors
        rgb_image = np.zeros((*selected_annotated_image.shape, 3), dtype=np.uint8)

        for region_id, rgb in selected_rgb_map.items():
            mask = selected_annotated_image == region_id
            rgb_image[mask] = rgb


        # Load the CSV file with probe cluster information
        #rel_path = os.path.relpath(os.path.dirname(probe_csv_file), results_folder_name)
        #save_path = os.path.join("Probe_on_brain_atlas_new", rel_path)
        #os.makedirs(save_path, exist_ok=True)
        save_path = save_folder + "/atlas"
        os.makedirs(save_path, exist_ok=True)
        # Load data from probe file and overlay it on the anatomy
        # Load probe positions and cluster labels
        probe_x_um = df['X (um)'].values
        probe_y_um = df['Y (um)'].values

        # calculate offsets in the x and y directions
        x_offset = annotation_shape[2]//2 # center at the midline of the brain
        y_offset = int(np.max(probe_y_um)/resolution_um)

        # flip y_values to match atlas direction
        probe_y_um = np.flip(probe_y_um)
        cluster_labels = df['Cluster Label'].values
        #cluster_labels = np.flip(cluster_labels)

        # Convert probe coordinates (um) to atlas pixel coordinates
        probe_x_px =  x_offset + slice_x + (probe_x_um / resolution_um)
        probe_y_px = rgb_image.shape[2] - y_offset - slice_y + (probe_y_um / resolution_um)

        if isinstance(colors, bool):
            # Create blue shades mapping going darker from top clusters to bottom clusters
            cluster_labels, sorted_labels = remap_labels(cluster_labels, probe_params)
            unique_labels = np.unique(cluster_labels)
            cluster_to_color, colors = cluster_color_map_for_labels(cluster_labels, cmap_name='Blues', min_shade=0.3)
        else:
            remapped_labels = np.zeros_like(cluster_labels)
            for new_label, old_label in enumerate(sorted_labels):
                remapped_labels[cluster_labels == old_label] = new_label + 1  # Start from 1
            cluster_labels = remapped_labels
            unique_labels = np.unique(cluster_labels)
            cluster_to_color, colors = cluster_color_map_for_labels(cluster_labels, cmap_name='Blues', min_shade=0.3)
            
        # Create figure with blue shaded clusters on the probe geometry
        # Create deterministic blue mapping for clusters using shared utility
        fig, ax = plt.subplots(figsize=(8,8))
        n_ch_per_shank = probe_params['n_ch_per_shank']
        n_cols = probe_params['n_cols']
        reshaped_labels = reshape(cluster_labels, n_ch_per_shank, n_cols, n_shanks)
        # add 2 empty columns between shanks for better visualization
        reshaped_labels = np.insert(reshaped_labels, np.arange(1, n_shanks) * n_cols, np.nan, axis=1)
        reshaped_labels_colored = np.array([[cluster_to_color[label] if not np.isnan(label) else (1,1,1,0) for label in row_label] for row_label in reshaped_labels])

        ax.set_facecolor('white')  # ensure axis background is white
        im = ax.imshow(reshaped_labels_colored, aspect='equal', origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'probe_cluster_labels_layout_{x}.{custom_format}'), format=custom_format, dpi=600, bbox_inches='tight')
        plt.close()

        # Plot anatomy and probe clusters
        text_pos = (int(0.12*annotation_shape[1]), int(0.72*annotation_shape[0]))  # position where to display the stereotactic coordinates

        if split_probes:
            if x == 0:
                plot_probe_on_atlas(rgb_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_labelled_atlas_original_pos_{x}.{custom_format}'), resolution_um, text_pos, save_to_pickle=True, custom_format=custom_format)
                plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_reference_atlas_original_pos_{x}.{custom_format}'), resolution_um, text_pos, save_to_pickle=True, custom_format=custom_format)
            else:
                plot_probe_on_atlas(rgb_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_labelled_atlas_original_pos_{x}.{custom_format}'), resolution_um, text_pos, stacked=True, custom_format=custom_format)
                plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_reference_atlas_original_pos_{x}.{custom_format}'), resolution_um, text_pos, stacked=True, custom_format=custom_format)
        else:
            plot_probe_on_atlas(rgb_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_labelled_atlas_original_pos.{custom_format}'), resolution_um, text_pos, custom_format=custom_format)
            plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px, probe_y_px, probe_params, np.round(AP,1), np.round(ML,1), np.round(DV,1), os.path.join(save_path, f'probe_reference_atlas_original_pos.{custom_format}'), resolution_um, text_pos, custom_format=custom_format)

        # find the best match in the x, y and z position to minimize the variation within clusters
        n_pixel_region = np.zeros(len(unique_labels))
        label_regions = np.zeros(len(unique_labels))
        x_shift = np.arange(-30, 30) # x shift range in pixels
        y_shift = np.arange(-30, 30) # y shift range in pixels 
        z_shift = np.arange(-8, 8) # z shift on the coronal plane
        shape = [len(x_shift), len(y_shift), len(z_shift)]

        evaluation_mat_label = np.zeros(shape) # evaluation matrix to check where the cluster match best the anatomy

        annotated_image = selected_atlas[slice_z , :, :]

        # Assign to each region the mode value of where it was originally placed
        for k, label in enumerate(unique_labels):
            label_x_px = probe_x_px[cluster_labels == label].astype(np.int16) 
            label_y_px = probe_y_px[cluster_labels == label].astype(np.int16)
            values_label = annotated_image[label_y_px, label_x_px]
            label_regions[k] = statistics.mode(values_label)

        for m, z_s in enumerate(z_shift):
            selected_reference_image = reference_image[slice_z + z_s, :,:]
            annotated_image = selected_atlas[slice_z + z_s, :, :]
            for i, x_s in enumerate(x_shift):
                for j, y_s in enumerate(y_shift):
                    for k, label in enumerate(unique_labels):
                        # if there is only one shank, only consider shifts that keep the probe in the same hemisphere
                        if n_shanks == 1:
                            if (probe_x_px[cluster_labels == label].mean() < annotation_shape[2]//2 and probe_x_px[cluster_labels == label].mean() + x_s > annotation_shape[2]//2) or (probe_x_px[cluster_labels == label].mean() > annotation_shape[2]//2 and probe_x_px[cluster_labels == label].mean() + x_s < annotation_shape[2]//2):
                                continue
                        label_x_px = probe_x_px[cluster_labels == label].astype(np.int16) + x_s
                        label_y_px = probe_y_px[cluster_labels == label].astype(np.int16) + y_s
                        values_label = annotated_image[label_y_px, label_x_px]

                        n_pixel_region[k] = np.sum(values_label==label_regions[k] )

                    evaluation_mat_label[i, j, m] = np.sum(n_pixel_region)


        # Find the indices of the minimum value in the evaluation matrix
        max_ind = np.unravel_index(np.argmax(evaluation_mat_label), evaluation_mat_label.shape)
        best_x_shift = x_shift[max_ind[0]]
        best_y_shift = y_shift[max_ind[1]]
        best_z_shift = z_shift[max_ind[2]]

        print("Best shift:", best_x_shift * resolution_um, best_y_shift * resolution_um, best_z_shift * resolution_um)
        print("N clusters in appropriate region = ",evaluation_mat_label[max_ind[0], max_ind[1], max_ind[2]] )


        plot_extent = [x_shift[0]*resolution_um, x_shift[-1]*resolution_um, y_shift[0]*resolution_um, y_shift[-1]*resolution_um]

        plt.figure()
        for j in range(len(z_shift)):
            plt.subplot(int(np.ceil(len(z_shift)/np.sqrt(len(z_shift)))), int(np.floor(np.sqrt(len(z_shift)))), j+1)
            plt.imshow( evaluation_mat_label[:,:,j].T, extent=plot_extent, vmin = np.min(evaluation_mat_label), vmax= np.max(evaluation_mat_label), origin='lower')
            # reduce size of x and y ticks
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            plt.colorbar(fraction=0.046, pad=0.04)
            # reduce size of colorbar ticks
            cbar = plt.gcf().axes[-1]
            cbar.tick_params(labelsize=5)
            plt.tight_layout()
            # plt.xlabel('X shift (um)')
            # plt.ylabel('Y shift (um)')
            # plt.title(f'Z shift: {z_shift[j]*resolution_um} um')
            if j == max_ind[2]:
                plt.plot(best_x_shift*resolution_um, best_y_shift*resolution_um, 'ro')
        # use a single colorbar for all subplots to the right of the figure
        
        plt.savefig(os.path.join(save_path, f'evaluation_matrix_repositioning_{x}.{custom_format}'), dpi=600, bbox_inches='tight')

        plt.figure(figsize=(8,8))
        plt.imshow(evaluation_mat_label[:,:,max_ind[2]].T, extent=plot_extent, origin='lower')
        plt.plot(best_x_shift*resolution_um, best_y_shift*resolution_um, 'ro')


        selected_annotated_image = selected_atlas[slice_z + best_z_shift, :, :]

        # Map annotation IDs to RGB colors
        rgb_image_sel = np.zeros((*selected_annotated_image.shape, 3), dtype=np.uint8)

        for region_id, rgb in selected_rgb_map.items():
            mask = selected_annotated_image == region_id
            rgb_image_sel[mask] = rgb


        selected_reference_image = reference_image[slice_z + best_z_shift,:,:]

        # Plot anatomy and probe clusters in adjusted position
        AP_new = np.round(AP - best_z_shift*resolution_um/1000,1)
        ML_new = np.round(ML - best_x_shift * resolution_um/1000, 1)
        DV_new = np.round(DV - best_y_shift*resolution_um/1000, 1)

        if split_probes:
            if x == 0:
                # save pickle object
                plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_reference_atlas_adjusted_{x}.{custom_format}'), resolution_um, text_pos, save_to_pickle=True, custom_format=custom_format)
                plot_probe_on_atlas(rgb_image_sel, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_original_label_atlas_adjusted_{x}.{custom_format}'), resolution_um, text_pos, save_to_pickle=True, custom_format=custom_format)
            else:
                # generate extra stacked plot
                plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_reference_atlas_adjusted_{x}.{custom_format}'), resolution_um, text_pos, stacked=True, custom_format=custom_format)
                plot_probe_on_atlas(rgb_image_sel, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_original_label_atlas_adjusted_{x}.{custom_format}'), resolution_um, text_pos, stacked=True, custom_format=custom_format)
        else:
            plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_reference_atlas_adjusted.{custom_format}'), resolution_um, text_pos, custom_format=custom_format)
            plot_probe_on_atlas(rgb_image_sel, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_original_label_atlas_adjusted.{custom_format}'), resolution_um, text_pos, custom_format=custom_format)

        plt.close('all')

    # selected_reference_image, full_labels, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, resolution_um, text_pos
    return (selected_reference_image, full_labels, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, probe_params, AP_new, ML_new, DV_new, resolution_um, text_pos)
