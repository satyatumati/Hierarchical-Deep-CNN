#!/bin/sh

num_classes=$1
num_train=$2
num_valid=$3
src_directory=$4
train_directory=$5
valid_directory=$6

echo $train_directory
echo $valid_directory

cd $src_directory

src_directory+="/*"

i=0
# glob for the files starting with b
for b in $src_directory; do 
   # test how many times the loop has been run and if it's less than 4...
   (( i++ < $num_classes )) && 
     # ... then move the files*
     
     base=$(basename $b)
     train="$train_directory"/"$base"
     valid="$valid_directory"/"$base"
     mkdir $train
     mkdir $valid

      cd "$b"
 
     mv  `ls | head -$num_train` $train
      mv  `ls | head -$num_valid` $valid
    
 cd ..
done

