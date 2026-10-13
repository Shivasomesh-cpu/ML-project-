import os
import numpy as np
import pandas as pd
import nibabel as nib

def create_mock_nifti(file_path, shape=(32, 32, 32)):
    """Create a small valid NIfTI file for testing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Synthetic brain-like volume (sphere in background)
    data = np.zeros(shape, dtype=np.float32)
    c = tuple(s // 2 for s in shape)
    r = min(shape) // 3
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = ((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2) <= r**2
    data[mask] = np.random.uniform(0.5, 1.5, size=np.count_nonzero(mask))
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)
    return file_path


def generate_mock_dataset(base_dir, n_subjects=12):
    """
    Generates realistic directory structure and metadata for PPMI, HBN, and NKI datasets.
    Allows end-to-end local testing of feature extraction, cross-validation, and analysis.
    """
    os.makedirs(base_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    # -------------------------------------------------------------
    # 1. PPMI Dataset Mock
    # -------------------------------------------------------------
    ppmi_dir = os.path.join(base_dir, "PPMI")
    ppmi_cat12_dir = os.path.join(ppmi_dir, "cat12")
    ppmi_turboprep_dir = os.path.join(ppmi_dir, "turboprep")
    os.makedirs(ppmi_cat12_dir, exist_ok=True)
    os.makedirs(ppmi_turboprep_dir, exist_ok=True)

    ppmi_subjects = [str(3000 + i) for i in range(n_subjects)]
    ppmi_sex = ['F' if i % 2 == 0 else 'M' for i in range(n_subjects)]
    ppmi_group = ['PD' if (i // 2) % 2 == 0 else 'Control' for i in range(n_subjects)]
    ppmi_age = [rng.uniform(50, 75) for _ in range(n_subjects)]

    ppmi_labels_df = pd.DataFrame({
        'PATNO': ppmi_subjects,
        'Sex': ppmi_sex,
        'Group': ppmi_group,
        'Age': ppmi_age
    })
    ppmi_labels_path = os.path.join(ppmi_dir, "ppmi_labels.csv")
    ppmi_labels_df.to_csv(ppmi_labels_path, index=False)

    # PPMI FreeSurfer CSVs
    cth_cols = ['PATNO', 'EVENT_ID'] + [f'lh_region_{i}_thickness' for i in range(34)] + [f'rh_region_{i}_thickness' for i in range(34)]
    sa_cols  = ['PATNO', 'EVENT_ID'] + [f'lh_region_{i}_area' for i in range(34)] + [f'rh_region_{i}_area' for i in range(34)]

    cth_rows, sa_rows = [], []
    for s in ppmi_subjects:
        cth_vals = [s, 'BL'] + list(rng.uniform(2.0, 3.5, size=68))
        sa_vals  = [s, 'BL'] + list(rng.uniform(1000, 3000, size=68))
        cth_rows.append(cth_vals)
        sa_rows.append(sa_vals)

    pd.DataFrame(cth_rows, columns=cth_cols).to_csv(os.path.join(ppmi_cat12_dir, "FS7_APARC_CTH_23Oct2025.csv"), index=False)
    pd.DataFrame(sa_rows, columns=sa_cols).to_csv(os.path.join(ppmi_cat12_dir, "FS7_APARC_SA_23Oct2025.csv"), index=False)

    # PPMI NIfTI files
    for s in ppmi_subjects:
        # CAT12 VBM
        cat12_file = os.path.join(ppmi_cat12_dir, f"sub-{s}", "anat", f"mwp1_{s}.nii")
        create_mock_nifti(cat12_file)
        # TurboPrep
        turboprep_file = os.path.join(ppmi_turboprep_dir, s, "normalized.nii.gz")
        create_mock_nifti(turboprep_file)

    # -------------------------------------------------------------
    # 2. HBN Dataset Mock
    # -------------------------------------------------------------
    hbn_dir = os.path.join(base_dir, "HBN")
    hbn_bids_dir = os.path.join(hbn_dir, "HBN_BIDS")
    hbn_fs_dir = os.path.join(hbn_dir, "HBN_FreeSurfer", "freesurfer")
    hbn_turboprep_dir = os.path.join(hbn_dir, "turboprep")
    os.makedirs(hbn_bids_dir, exist_ok=True)
    os.makedirs(hbn_fs_dir, exist_ok=True)
    os.makedirs(hbn_turboprep_dir, exist_ok=True)

    hbn_subjects = [f"NDAR{1000 + i}" for i in range(n_subjects)]
    hbn_sex = ['Female' if i % 2 == 0 else 'Male' for i in range(n_subjects)]
    hbn_age = [rng.uniform(8, 18) for _ in range(n_subjects)]

    hbn_demo_df = pd.DataFrame({
        'participant_id': hbn_subjects,
        'sex': hbn_sex,
        'age': hbn_age
    })
    hbn_demo_file = os.path.join(hbn_bids_dir, "final_preprocessed_subjects_with_demographics.tsv")
    hbn_demo_df.to_csv(hbn_demo_file, sep='\t', index=False)

    for s in hbn_subjects:
        # CAT12
        cat12_file = os.path.join(hbn_bids_dir, f"sub-{s}", "ses-1", "anat", "mri", f"mwp1sub-{s}.nii")
        create_mock_nifti(cat12_file)
        # TurboPrep
        turboprep_file = os.path.join(hbn_turboprep_dir, s, "normalized.nii.gz")
        create_mock_nifti(turboprep_file)
        # FreeSurfer surface stats (Schaefer 400 + aparc 68)
        schaefer_names = [f"7Networks_LH_Vis_{i+1}" for i in range(200)] + [f"7Networks_RH_Vis_{i+1}" for i in range(200)]
        schaefer_hemi = ["lh"] * 200 + ["rh"] * 200
        df_schaefer = pd.DataFrame({
            'atlas': ["Schaefer2018_400Parcels_17Networks_order"] * 400,
            'StructName': schaefer_names,
            'SurfArea': rng.uniform(800, 2500, size=400),
            'ThickAvg': rng.uniform(2.1, 3.4, size=400),
            'hemisphere': schaefer_hemi
        })
        aparc_names = [f"aparc_lh_{i}" for i in range(34)] + [f"aparc_rh_{i}" for i in range(34)]
        aparc_hemi = ["lh"] * 34 + ["rh"] * 34
        df_aparc = pd.DataFrame({
            'atlas': ["aparc"] * 68,
            'StructName': aparc_names,
            'SurfArea': rng.uniform(1000, 3000, size=68),
            'ThickAvg': rng.uniform(2.0, 3.5, size=68),
            'hemisphere': aparc_hemi
        })
        fs_table = pd.concat([df_schaefer, df_aparc], ignore_index=True)
        s_fs_dir = os.path.join(hbn_fs_dir, f"sub-{s}")
        os.makedirs(s_fs_dir, exist_ok=True)
        fs_table.to_csv(os.path.join(s_fs_dir, f"sub-{s}_regionsurfacestats.tsv"), sep='\t', index=False)

    # -------------------------------------------------------------
    # 3. NKI Dataset Mock
    # -------------------------------------------------------------
    nki_dir = os.path.join(base_dir, "NKI")
    nki_bids_dir = os.path.join(nki_dir, "NKI_BIDS")
    nki_fs_dir = os.path.join(nki_dir, "NKI_FreeSurfer", "freesurfer")
    nki_turboprep_dir = os.path.join(nki_dir, "turboprep_NKI")
    os.makedirs(nki_bids_dir, exist_ok=True)
    os.makedirs(nki_fs_dir, exist_ok=True)
    os.makedirs(nki_turboprep_dir, exist_ok=True)

    nki_subjects = [f"NKI_{2000 + i}" for i in range(n_subjects)]
    nki_sex = ['F' if i % 2 == 0 else 'M' for i in range(n_subjects)]
    nki_age = [rng.uniform(20, 70) for _ in range(n_subjects)]

    nki_demo_file = os.path.join(nki_dir, "final_demographics.txt")
    with open(nki_demo_file, "w") as f:
        for s, sx, ag in zip(nki_subjects, nki_sex, nki_age):
            f.write(f"{s}\t{sx}\t{ag:.2f}\n")

    for s in nki_subjects:
        # CAT12
        cat12_file = os.path.join(nki_bids_dir, f"sub-{s}", "ses-BAS1", "anat", "mri", f"mwp1_{s}.nii")
        create_mock_nifti(cat12_file)
        # TurboPrep
        turboprep_file = os.path.join(nki_turboprep_dir, s, "normalized.nii.gz")
        create_mock_nifti(turboprep_file)
        # FreeSurfer surface stats
        s_fs_dir = os.path.join(nki_fs_dir, f"sub-{s}_ses-BAS1")
        os.makedirs(s_fs_dir, exist_ok=True)
        fs_table.to_csv(os.path.join(s_fs_dir, f"sub-{s}_ses-BAS1_regionsurfacestats.tsv"), sep='\t', index=False)

    paths = {
        'PPMI': {
            'labels_path': ppmi_labels_path,
            'cat12_dir': ppmi_cat12_dir,
            'turboprep_dir': ppmi_turboprep_dir
        },
        'HBN': {
            'bids_dir': hbn_bids_dir,
            'demo_file': hbn_demo_file,
            'fs_dir': hbn_fs_dir,
            'turboprep_dir': hbn_turboprep_dir
        },
        'NKI': {
            'base_dir': nki_dir,
            'bids_dir': nki_bids_dir,
            'demo_file': nki_demo_file,
            'fs_dir': nki_fs_dir,
            'turboprep_dir': nki_turboprep_dir
        }
    }
    print(f"Mock datasets generated successfully in '{base_dir}' with {n_subjects} subjects per dataset.")
    return paths
