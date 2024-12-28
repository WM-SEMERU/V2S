#! /bin/bash

for d in ~/Downloads/study-videos-new-model/*/ ; do
    name=$(basename $d)
    for e in $d* ; do
        app=$(basename $e)
        cp $e/detection_new_model/detection.json ../user_data/$name/$app/detection.json
    done
done