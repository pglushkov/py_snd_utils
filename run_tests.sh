
# some variables ...
emphasis_data_dir=~/OPIZDENEY/__EMPHASIS/short_db_for_auto_masking
input_dir=$emphasis_data_dir/renamed_orig
mask_dir=$emphasis_data_dir/renamed_manual_mask
output_dir=tmp/emph_detect
if_core=angry1
input_file=$input_dir/$if_core".wav"
mask_file=$mask_dir/$if_core"_proc.wav"
output_file=$if_core".wav"

# 1. run emphasis-detector on a single file
#python3 run_emphasis_detector.py --mode file -i $input_file -o $output_file -m $mask_file


# 2. run emphasis-detector on a directory
python3 run_emphasis_detector.py --mode dir -i $input_dir -o $output_dir -m $mask_dir
