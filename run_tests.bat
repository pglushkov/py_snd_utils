rem echo off

rem some variables ...
set emphasis_data_dir=./../short_db_for_auto_masking
set input_dir=%emphasis_data_dir%/renamed_orig
set mask_dir=%emphasis_data_dir%/renamed_manual_mask
set output_dir=tmp/emph_detect
set if_core=angry1
set input_file=%input_dir%/%if_core%".wav"
set mask_file=%mask_dir%/%if_core%"_proc.wav"
set output_file=%if_core%".wav"

rem 1. run emphasis-detector on a single file
rem python run_emphasis_detector.py --mode file -i %input_file% -o %output_file% -m %mask_file%


rem 2. run emphasis-detector on a directory
python run_emphasis_detector.py --mode dir -i %input_dir% -o %output_dir% -m %mask_dir%
