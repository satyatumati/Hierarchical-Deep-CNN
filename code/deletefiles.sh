#!/bin/sh
for d in */ ; do
   
    # cd $d
   
   
    if [ $(find $d -type f | wc -l)  -lt 20 ] ; 
    then
     
      rm -rf $d
    fi
   
  #  cd ..
done
